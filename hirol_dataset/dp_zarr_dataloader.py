from reader import ActionType, Action_Type_Mapping_Dict, ObservationType
from data_loader import DataLoaderBase
import os, warnings, sys
import numpy as np
import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import glog as log
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import tempfile
import shutil

def process_episode_worker(args):
    """Worker function for processing a single episode in parallel"""
    config, task_dir, episode_dir, temp_dir = args

    try:
        # Create a temporary DiffusionPolicyLoader instance for this process
        temp_zarr_path = os.path.join(temp_dir, f"temp_{os.getpid()}_{episode_dir}.zarr")

        # Create temporary config with unique output path
        temp_config = config.copy()
        temp_config['output_path'] = temp_dir
        temp_config['data_name'] = f"temp_{os.getpid()}_{episode_dir}"

        loader = DiffusionPolicyLoader(temp_config, task_dir)
        loader.create_zarr_dataset(task_dir, episode_dir)

        return temp_zarr_path

    except Exception as e:
        log.error(f"Error processing episode {episode_dir}: {str(e)}")
        return None

class DiffusionPolicyLoader(DataLoaderBase):
    def __init__(self, config, task_dir, json_file_name = "data.json", action_type = ActionType.JOINT_POSITION,
                 observation_type = ObservationType.JOINT_POSITION_ONLY):
        action_type = config.get("action_type", "joint_position")
        if action_type not in Action_Type_Mapping_Dict:
            raise ValueError(f'{action_type} is not supported!!')
        action_type = Action_Type_Mapping_Dict[action_type]
        json_file_name = config.get(f'json_file_name', "data.json")
        super().__init__(config, task_dir, json_file_name, action_type, observation_type)
        
        self._task_list = config.get('task_list', None)
        if self._task_list is None:
            self._task_list = [task_dir]
        else:
            for i, task_name in enumerate(self._task_list):
                self._task_list[i] = os.path.join(self._task_dir, task_name)
        for task_dir in self._task_list:
            if not os.path.exists(task_dir):
                raise ValueError(f'Could not find {task_dir}')
        log.info(f'task list: {self._task_list}')
        
        self._orientation_representation = 'euler'
        self._output_path_dir = config.get('output_path', './data')
        self._file_name = config.get('data_name', 'dataset')
        self._file_name += '.zarr'
        self._zarr_path = os.path.join(self._output_path_dir, self._file_name)
        self._load_fps = config.get("fps", 30)
        
        # replay buffer
        file_exists = os.path.exists(self._zarr_path)
        mode = 'a' if file_exists else 'w'
        self._replay_buffer = ReplayBuffer.create_from_path(self._zarr_path, mode=mode)
        
    def put_key_value_to_zarr_data(self, zarr_data, key, value):
        appended_data = np.array(value)[None]
        # log.info(f'{key}, shape: {appended_data.shape}')
        if key not in zarr_data:
            zarr_data[key] = appended_data
        else:
            zarr_data[key] = np.vstack((zarr_data[key], appended_data))
    
    def create_zarr_dataset(self, task_dir, episode_dir):
        n_episodes = self._replay_buffer.n_episodes
        curr_episode_id = n_episodes

        # episode data appending
        skip_num_step = int(30 / self._load_fps)
        episode_data, _ = self.load_episode(task_dir, episode_dir, skip_num_step)
        log.info(f'Successfully load the {episode_dir} from {task_dir}')
        if episode_data is None:
            log.warn(f'{task_dir}/{episode_dir} has problem during loading the episode data')
            return
        
        zarr_episode_data = {}
        episode_step_nums = 0
        for stpes, step_data in tqdm(enumerate(episode_data), desc="processing steps ", unit="steps"):
            # images
            colors_data = step_data.get("colors", {})
            for key, value in colors_data.items():
                self.put_key_value_to_zarr_data(zarr_episode_data, key, value)
            
            # state and action and stages
            raw_obs_states = step_data.get("observations", {})
            # log.info(f'raw obs {raw_obs_states}')
            actions = step_data.get("actions", {})
            obs_state = np.array([]); zarr_actions = np.array([]); zarr_stages = np.array([])
            for id, (key, value) in enumerate(raw_obs_states.items()):
                obs_state = np.hstack((obs_state, value))
                zarr_actions = np.hstack((zarr_actions, actions[key]))
                zarr_stages = np.hstack((zarr_stages, [1]))
            self.put_key_value_to_zarr_data(zarr_episode_data, "state", obs_state)
            self.put_key_value_to_zarr_data(zarr_episode_data, "action", zarr_actions)
            self.put_key_value_to_zarr_data(zarr_episode_data, "stage", zarr_stages)
            episode_step_nums += 1
        self._replay_buffer.add_episode(zarr_episode_data)
        # log.info(f'state shape: {zarr_episode_data["state"].shape}')
        # log.info(f'The episode data {curr_episode_id} contains {episode_step_nums} and is successfully save to {self._zarr_path} from {self._task_dir}/{episode_dir}')
        # log.debug(f'episode_data: {zarr_episode_data}')

    def convert_dataset_parallel(self, num_processes=None):
        """Parallel version of convert_dataset using multiprocessing"""
        if num_processes is None:
            num_processes = mp.cpu_count()

        log.info(f"Using {num_processes} processes for parallel processing")

        # Create temporary directory for intermediate zarr files
        temp_dir = tempfile.mkdtemp(prefix="zarr_temp_")
        log.info(f"Created temporary directory: {temp_dir}")

        try:
            all_temp_paths = []

            for task_dir in tqdm(self._task_list, desc="processing tasks", unit="tasks"):
                episode_dirs = [d for d in os.listdir(task_dir) if 'episode' in d]
                log.info(f"Found {len(episode_dirs)} episodes in {task_dir}")

                if not episode_dirs:
                    continue

                # Prepare arguments for worker processes
                args_list = [(self._config, task_dir, episode_dir, temp_dir)
                           for episode_dir in episode_dirs]

                # Process episodes in parallel
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    temp_paths = list(tqdm(
                        executor.map(process_episode_worker, args_list),
                        total=len(args_list),
                        desc=f"processing episodes in {os.path.basename(task_dir)}"
                    ))

                # Filter out None results (failed episodes)
                valid_temp_paths = [path for path in temp_paths if path is not None]
                all_temp_paths.extend(valid_temp_paths)

                log.info(f'Finished processing {task_dir}: {len(valid_temp_paths)}/{len(episode_dirs)} episodes successful')

            # Merge all temporary zarr files
            log.info(f"Merging {len(all_temp_paths)} temporary zarr files...")
            self.merge_zarr_files(all_temp_paths)

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                log.info(f"Cleaned up temporary directory: {temp_dir}")

        log.info(f'Created all episodes in {self._task_list} with parallel processing')

    def convert_dataset(self):
        """Original serial version of convert_dataset"""
        for task_dir in tqdm(self._task_list, desc="processing tasks", unit="tasks"):
            episode_dirs = os.listdir(task_dir)
            for episode_dir in tqdm(episode_dirs, desc="processing episodes ", unit="episodes"):
                self.create_zarr_dataset(task_dir, episode_dir)
            log.info(f'Finished processing the {task_dir}')
        log.info(f'created all episodes in {self._task_list}')

    def merge_zarr_files(self, temp_zarr_paths):
        """Merge multiple temporary zarr files into the main zarr file"""
        for temp_path in temp_zarr_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    temp_buffer = ReplayBuffer.create_from_path(temp_path, mode='r')
                    for episode_idx in range(temp_buffer.n_episodes):
                        episode_data = temp_buffer.get_episode(episode_idx)
                        self._replay_buffer.add_episode(episode_data)
                    log.info(f"Merged {temp_path} with {temp_buffer.n_episodes} episodes")
                    # Clean up temporary file
                    shutil.rmtree(temp_path)
                except Exception as e:
                    log.error(f"Error merging {temp_path}: {str(e)}")

if __name__ == "__main__":
    import yaml
    task_list = [{'name': "insert_tube", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/dp_zarr_insert_tube_config.yaml'},
                 {'name': "insert_tube_jps2pose_euler", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/dp_zarr_insert_tube_jps2pose_euler_config.yaml'},
                 {'name': "insert_tube_jps2delPose_euler", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/dp_zarr_insert_tube_jps2delPose_euler_config.yaml'},
                 {'name': "insert_tube_pose2delPose_euler", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/dp_zarr_insert_tube_pose2delPose_euler_config.yaml'},
                 {'name': "pick_n_place", 'task_dir': '/boot/common_data/2025/fr3/1015_pick_and_place_fr3_3dmouse_250ep', 
                  'cfg_file': 'config/dp_zarr_pick_n_place_config.yaml'},
                 {'name': "block_stacking", 'task_dir': '/boot/common_data/2025/fr3/1018_block_stacking_fr3_3Dmosue_110eps', 
                  'cfg_file': 'config/dp_zarr_block_satcking_config.yaml'},
                ]
    
    num_task = len(task_list)
    for i, task in enumerate(task_list):
        print(f'{"=="*5} {i}th task name: {task["name"]}, task dir: {task["task_dir"]}, cfg: {task["cfg_file"]} {"=="*5}')
    get_task = False
    key = -1
    while not get_task:
        key = input(f'Please select the task from 0 - {num_task - 1}: ')
        key = int(key)
        if key >= 0 and key < num_task:
            get_task = True
    print(f'Selected {task_list[key]["name"]} to convert data!!!!')
    
    config_file = task_list[key]["cfg_file"]
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, config_file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'config: {config}')
    task_dir = task_list[key]["task_dir"]
    action_type = config["action_type"]
    action_type = Action_Type_Mapping_Dict[action_type]
    obs_type = config["obs_type"]
    obs_type = ObservationType(obs_type)

    dp_data_loader = DiffusionPolicyLoader(config, task_dir, action_type=action_type,
                                           observation_type=obs_type)

    # Check if parallel processing is enabled in config
    use_parallel = config.get('parallel', False)
    num_processes = config.get('num_processes', None)

    if use_parallel:
        log.info(f"Using parallel processing with {num_processes or mp.cpu_count()} processes")
        dp_data_loader.convert_dataset_parallel(num_processes=num_processes)
    else:
        log.info("Using serial processing")
        dp_data_loader.convert_dataset()

    # loading test
    dataset_path = dp_data_loader._zarr_path
    replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    log.info(f'total episode: {replay_buffer.n_episodes}')
    log.info(f'repaly buffer data: {replay_buffer.data}')
    log.info(f'repaly buffer store: {replay_buffer.root.store}')
    
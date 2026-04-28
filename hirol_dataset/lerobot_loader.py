from reader import RerunEpisodeReader, ActionType
from data_loader import DataLoaderBase
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os, warnings
import numpy as np
from tqdm import tqdm
import glog as log
import cv2

class LerobotLoader(DataLoaderBase):
    def __init__(self, config, task_dir, json_file_name = "data.json", action_type = ActionType.JOINT_POSITION):
        super().__init__(config, task_dir, json_file_name, action_type)
        self._push_to_repo = self._config.get("push_to_repo", False)
        self._robot_name = config.get("robot_name", "fr3")
        self._repo_name = config.get("repo_name", "peg_in_hole")
        self._task_list = config.get("task_list", None)
        self._output_root_path = config.get("root_path", "../assets/data")
        self._load_fps = config.get("fps", 10)
        # 一种是只给一个task的文件夹
        if self._task_list is None:
            self._task_list = [self._task_dir]
        else: # 另一种是个所有task的文件夹并给每个task的名字在list里
            for i, task in enumerate(self._task_list):
                cur_task_dir = os.path.join(self._task_dir, task)
                if not os.path.exists(cur_task_dir):
                    raise ValueError(f'{cur_task_dir} did not exist, please check the value of task dir and task list')
                self._task_list[i] = cur_task_dir
        
        
        
    def get_example_feature_dim(self, example_step):
        # try to confirm the images and state length,
        images = example_step.get("colors", {})
        depths = example_step.get("depths", {})
        self._image_keys = []; self._image_resolutions = []
        for key, image in images.items():
            self._image_keys.append(key)
            self._image_resolutions.append(tuple(image.shape))
        for key, image in depths.items():
            self._image_keys.append(key)
            self._image_resolutions.append(tuple(image.shape))
        print(f'image keys: {self._image_keys}')
        # @TODO: Concataneate the tool state into the end
        self._joint_states_dim = 0; self._action_dim = 0
        joint_states = example_step.get("joint_states", {})
        actions = example_step.get("actions", {})
        for key, state in joint_states.items():
            self._joint_states_dim += len(state["position"])
        for key, action in actions.items():
            self._action_dim += len(action)
        
        self._tool_state_dim = 0
        tool_states = example_step.get("tools", {})
        for key, tool_state in tool_states.items():
            cur_tool_state_value = tool_state["position"]
            if isinstance(cur_tool_state_value, list):
                self._tool_state_dim += len(tool_state["position"])
            else: self._tool_state_dim += 1 # for floating number case
        self._total_obs_state_dim = self._joint_states_dim + self._tool_state_dim
        print(f'total obs dim: {self._total_obs_state_dim}, joint: {self._joint_states_dim}, tool: {self._tool_state_dim}')
        print(f'action dim: {self._action_dim}')
    
    def convert_dataset(self):
        skip_nums_steps = int(30.0 / self._load_fps)
        task_dir = self._task_list[0]
        episode_dir = os.listdir(task_dir)[0]
        example_episode, _ = self.load_episode(task_dir, episode_dir, skip_nums_steps)
        self.get_example_feature_dim(example_episode[0])
        feature_dicts = {}
        for i, key in enumerate(self._image_keys):
            feature_dicts[key] = dict(dtype = "image",
                                shape = self._image_resolutions[i],
                    names = ["height", "width", "channel"],)
        feature_dicts["state"] = {"dtype": "float64",
                                  "shape": (self._total_obs_state_dim,),
                                  "names": ["state"],}
        feature_dicts["actions"] = {"dtype": "float64",
                                    "shape": (self._action_dim,),
                                    "names": ["actions"],}
        
        save_path = os.path.join(self._output_root_path, self._repo_name)
        self._lerobot_dataset = LeRobotDataset.create(
            root= save_path,
            repo_id=self._repo_name,
            robot_type=self._robot_name,
            fps=self._load_fps,
            features=feature_dicts,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        
        state_dismatch_list = []; action_dismatch_list = []
        state_step_list = []; action_step_list = []; 
        # 每一个task下的所有episodes
        for i, task_dir in tqdm(enumerate(self._task_list), desc=f"processing tasks", unit="task"):
            # text_info = self._all_episode_text[i]
            dirs = os.listdir(task_dir)
            # 同一个task下的每一个episode
            for cur_episode_dir in tqdm(dirs, desc=f"processing episodes", unit="episode"):
                episode_data, text_info = self.load_episode(task_dir, cur_episode_dir, skip_nums_steps)
                if episode_data is None:
                    continue
                state_wrong_nums = 0; action_wrong_nums = 0 
                for i, step in tqdm(enumerate(episode_data), desc=f"processing steps", unit="step"):
                    frame_feature = {}
                    for image_key in self._image_keys:
                        step_key = "colors" if "color" in image_key else "depths"
                        # print(f'step_key: {step_key}, image_key: {image_key} step: {step[step_key]}')
                        # print(f'{i}th {step["idx"]}')
                        # cv2.imshow(f'{step_key}_{image_key}', step[step_key][image_key])
                        # cv2.waitKey(1)
                        frame_feature[image_key] = step[step_key][image_key]
                    frame_feature["state"] = np.array([])
                    tool_states = step["tools"]
                    # state: [arm state, tool state, another_arm_state, another_tool_state]
                    for key, value in step["joint_states"].items():
                        cur_feature_state = np.hstack((value["position"], tool_states[key]["position"]))
                        frame_feature["state"] = np.hstack((frame_feature["state"], cur_feature_state))
                    print(f'state: {frame_feature["state"]}')
                    if len(frame_feature["state"]) != self._total_obs_state_dim:
                        warnings.warn(f'{task_dir} {episode_dir} has wrong state dim: {len(frame_feature["state"])} in {i}th step')
                        state_wrong = f'{task_dir}_{cur_episode_dir}'
                        if not state_wrong in state_dismatch_list:
                            state_dismatch_list.append(state_wrong)
                        state_wrong_nums += 1
                        continue
                    frame_feature["actions"] = np.array([])
                    for key, value in step["actions"].items():
                        frame_feature["actions"] = np.hstack((frame_feature["actions"], value))
                    print(f'actions: {frame_feature["actions"]}')
                    action_dim = len(frame_feature["actions"])
                    if action_dim != self._action_dim:
                        warnings.warn(f'{task_dir} {episode_dir} has wrong action dim: {action_dim} in {i}th step')
                        action_wrong = f'{task_dir}_{cur_episode_dir}'
                        if not action_wrong in state_dismatch_list:
                            action_dismatch_list.append(action_wrong)
                        action_wrong_nums += 1
                        continue
                    frame_feature["task"] = text_info
                    # print(f'frame_feature: {frame_feature}')
                    # raise ValueError
                    
                    self._lerobot_dataset.add_frame(frame=frame_feature)
                if state_wrong_nums != 0:
                    state_step_list.append(state_wrong_nums)
                if action_wrong_nums != 0:
                    action_step_list.append(action_wrong_nums)
                self._lerobot_dataset.save_episode()
                del episode_data
                
                log.info(f'Successfully processed {task_dir}/{cur_episode_dir}'
                         f'and saved to {save_path}')
                log.info(f'{len(self._lack_data_json_list)} lacks the data.json files: {self._lack_data_json_list}')
                log.info(f'state: {state_dismatch_list}, len: {state_step_list}')
                log.info(f'action: {action_dismatch_list}, len: {action_step_list}')
        cv2.destroyAllWindows()
            
        log.info(f'{len(self._lack_data_json_list)} lacks the data.json files: {self._lack_data_json_list}')
        log.info(f'state: {state_dismatch_list}, len: {state_step_list}')
        log.info(f'action: {action_dismatch_list}, len: {action_step_list}')
        
        # Optionally push to the Hugging Face Hub
        if self._push_to_repo:
            self._lerobot_dataset.push_to_hub(
                tags=[self._robot_name, self._repo_name],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )
        return self._lerobot_dataset
    
    
if __name__ == '__main__':
    import yaml
    config_file = "config/lerobot_config.yaml"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, config_file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'config: {config}')
    task_dir = "data/peg_in_hole"
   
    lerobot_dataset = LerobotLoader(config, task_dir)
    dataset = lerobot_dataset.convert_dataset()
    print(f'len dataset: {len(dataset)}')
    print(f'dataset: {dataset}')
    # lerobot自动转换为tensor
    data_sample = next(iter(dataset))
    print(f'data: {data_sample}')
    
    # config.get("root_path", "../assets/data")
    # import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    # root_path = config["root_path"]
    # repo_name = config["repo_name"]
    # root_path = os.path.join(root_path, repo_name)
    # dataset = lerobot_dataset.LeRobotDataset(repo_id=repo_name, root=root_path)
    # print(f'len dataset: {len(dataset)}')
    # print(f'dataset: {dataset}')
    # data_sample = next(iter(dataset))
    # print(f'data: {data_sample}')
    
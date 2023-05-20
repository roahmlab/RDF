import os
import subprocess
class VideoRecorder:
    def __init__(self,
        env,
        base_path,
        metadata=None,
        enabled=None,
        dpi=None,
        frame_rate=5,
        show=False,
        text=None,
        format = 'mp4'
        ):
        #assert format == 'mp4' or format == 'gif'
        self.env =env 
        head, _ = os.path.split(base_path)
        self.format = format
        self.video_path = base_path 
        self.frame_dir_path = os.path.join(head,'frames')
        self.save_kwargs = {'frame_rate':frame_rate,'save_path':self.frame_dir_path, 'dpi':dpi, 'text':text}
        self.show = show

    def capture_frame(self,FO_link=None):
        self.env.render(FO_link=FO_link,show=self.show,save_kwargs = self.save_kwargs) 

    def close(self,rm_frames=True):
        fr = str(2*self.save_kwargs['frame_rate'])
        subprocess.call([
            'ffmpeg',
            '-y', # overwrite 
            '-framerate',fr, # how often change to next frame of image
            '-i', os.path.join(self.frame_dir_path,'frame_%04d.png'), # frame images for video geneartion
            '-r', fr, # frame rate of video
            # '-pix_fmt', 'yuv420p',
            #'-crf','25', # quality of video, lower means better
            # '-vcodec','h264',
            '-s', '1280x960', # '1280x960', '640x480', '320x240'
            self.video_path + '.mp4'
        ],stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)   

        # for some reason, direct .gif generation is poor.
        if self.format == 'gif': 
            subprocess.call([
                'ffmpeg',
                '-y',
                '-i',self.video_path + '.mp4',
                self.video_path + '.gif'

            ],stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
            subprocess.call(['rm','-rf',self.video_path + '.mp4'])

        if rm_frames:
            subprocess.call(['rm','-rf',self.frame_dir_path])
        self.env.close()




if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    from zonopy.environments.arm_3d import Arm_3D
    from zonopy.environments.parallel_arm_2d import Parallel_Arm_2D
    from zonopy.environments.parallel_arm_3d import Parallel_Arm_3D
        
    import torch 
    import time 
    parallel = True
    if parallel:
        env = Parallel_Arm_3D(n_envs = 10, robot='Kinova3', n_obs=10, n_plots = 4,T_len=24)
        #env = Parallel_Arm_2D(n_links = 2 ,n_obs = 4,T_len=24,n_envs=10,n_plots=4)
        video_folder = 'video_test'

        ts = time.time()
        for i in range(1):
            base_path = os.path.join(video_folder,f'video_{i}')
            video_recorder = VideoRecorder(env,base_path,frame_rate=3,format='gif',show=False)
            for t in range(20):
                env.step(torch.rand(env.n_envs,env.n_links))
                video_recorder.capture_frame()
            video_recorder.close(True)
            env.reset()

        print(f'Time elasped: {time.time()-ts}')    

    else:
        #env = Arm_2D(n_links = 2 ,n_obs = 1,T_len=24)
        env = Arm_3D(n_obs = 2, T_len = 24)
        video_folder = 'video_test'

        ts = time.time()
        base_path = os.path.join(video_folder,f'video')
        for i in range(2):
            
            video_recorder = VideoRecorder(env,base_path,frame_rate=3,format='gif')
            for t in range(5):
                env.step(torch.rand(env.n_links))
                video_recorder.capture_frame()
            
            env.reset()
        video_recorder.close(True)
        print(f'Time elasped: {time.time()-ts}')

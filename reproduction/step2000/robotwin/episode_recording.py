"""Diagnostic episode recording only; never changes actions or advances physics."""
import json, subprocess
from pathlib import Path
import numpy as np

class EpisodeRecorder:
    def __init__(self, root, precision, setting, task, index):
        self.enabled = index == 0 and not (Path(root)/'recordings/local_quota_verified.json').exists()
        self.proc = None
        self.frames = 0
        self.last_step = None
        self.sampled_steps = []
        self.error = None
        self.path = Path(root)/'recordings'/precision/setting/task/f'{index:02d}.mp4'

    def capture(self, env, force=False):
        if not self.enabled or self.error:
            return
        step = int(env.take_action_cnt)
        if step == self.last_step or (not force and step % 10 != 0):
            return
        try:
            # Read cameras only: no get_obs, RNG, policy, reward or physics calls.
            env._update_render()
            env.cameras.update_picture()
            rgb = env.cameras.get_rgb()
            frame = np.ascontiguousarray(np.concatenate([rgb[k]['rgb'] for k in ('head_camera','left_camera','right_camera')], axis=1), dtype=np.uint8)
            if self.proc is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                h,w,_ = frame.shape
                self.log = self.path.with_suffix('.ffmpeg.log').open('w')
                self.proc = subprocess.Popen(['ffmpeg','-nostdin','-n','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE,stderr=self.log,stdout=subprocess.DEVNULL)
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1
            self.last_step = step
            self.sampled_steps.append(step)
        except Exception as e:
            self.error = repr(e)
            print('RECORDING_ERROR', self.error, flush=True)

    def finish(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                rc=self.proc.wait(timeout=60)
                if rc: self.error = self.error or f'ffmpeg exit {rc}'
            except Exception as e:
                self.error = repr(e)
                self.proc.kill()
                self.proc.wait()
            self.log.close()
        return dict(enabled=self.enabled,path=str(self.path) if self.proc else None,frames=self.frames,fps=10,cameras=['head_camera','left_camera','right_camera'],sampled_action_steps=self.sampled_steps,sampling='initial, every 10 policy actions, and terminal frame; playback is not wall-clock time',error=self.error,complete=bool(self.proc and not self.error))

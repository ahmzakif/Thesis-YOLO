import time

class SpeedTracker:
    def __init__(self):
        self.frame_count = 0
        self.total_latency = 0
        self.total_fps = 0
        self.highest_fps = 0
        self.fastest_latency = float('inf')

    def speed_performance(self, start_time, end_time, limit_frame=None):
        latency = end_time - start_time
        fps = 1 / latency

        self.total_latency += latency
        self.total_fps += fps
        self.frame_count += 1
        self.highest_fps = max(self.highest_fps, fps)
        self.fastest_latency = min(self.fastest_latency, latency)

        if limit_frame and self.frame_count >= limit_frame:
            print("\nPerformance Summary:")
            print(f"Average Latency: {self.total_latency / self.frame_count:.2f} seconds")
            print(f"Fastest Latency: {self.fastest_latency:.2f} seconds")
            print(f"Average FPS: {self.total_fps / self.frame_count:.2f}")
            print(f"Highest FPS: {self.highest_fps:.2f}")

            self.frame_count = 0
            self.total_latency = 0
            self.total_fps = 0
            self.highest_fps = 0
            self.fastest_latency = float('inf')
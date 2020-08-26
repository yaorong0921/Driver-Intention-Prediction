import random
import math


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

class UniformRandomSample(object):
    """Sample partially from each second in the given frame indices.

    If the number of frames is less than 25 (about one second),
    then take the first 16 framens and the last 16 frames.

    Args:
        size (int): Desired output size of the crop.
        end_sec (int): Video length (sec) of the clip
    """

    def __init__(self, size, end_sec):
        self.size = size
        self.end_sec = end_sec

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        out = []

        if len(frame_indices) < 25* self.end_sec:
            out.extend(list(range(len(frame_indices)-self.size,len(frame_indices))))
        else:
            sample_per_sec = self.size//self.end_sec
            sample_base = random.randint(0, 25-sample_per_sec-1)

            for i in range(0, self.end_sec-1):
                out.extend(list(range(sample_base+i*25, sample_base+i*25+sample_per_sec)))
            sample_rest = self.size-len(out)
            out.extend(list(range(sample_base+(self.end_sec-1)*25, sample_base+(self.end_sec-1)*25+sample_rest)))
        out_idx = []
        for idx in out:
            out_idx.append(frame_indices[idx])

        return out_idx

class UniformEndSample(object):
    """Sample partially from (END) of each second in the given frame indices.

    If the number of frames is less than 25 (about one second),
    then take the first 16 framens and the last 16 frames.

    Args:
        size (int): Desired output size of the crop.
        end_sec (int): Video length (sec) of the clip
    """

    def __init__(self, size, end_sec):
        self.size = size
        self.end_sec = end_sec

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        out = []

        if len(frame_indices) < 25*self.end_sec:
            out.extend(list(range(len(frame_indices)-self.size,len(frame_indices))))
        else:
            sample_per_sec = self.size//self.end_sec
            sample_base = 25-sample_per_sec-1
            for i in range(0, self.end_sec-1):
                out.extend(list(range(sample_base+i*25, sample_base+i*25+sample_per_sec)))
            sample_rest = self.size-len(out)
            out.extend(list(range(sample_base+(self.end_sec-1)*25, sample_base+(self.end_sec-1)*25+sample_rest)))
        out_idx = []
        for idx in out:
            out_idx.append(frame_indices[idx])

        return out_idx

class UniformIntervalCrop(object):
    def __init__(self, size, interval):
        self.size = size
        self.interval = interval

    def __call__(self, frame_indices):
        last_input = frame_indices[-1]
        target = last_input
        out = [last_input-self.interval+1]

        while (len(out)<self.size):
            out.append(out[-1]-self.interval+1)
        out.reverse()

        return out, [target]
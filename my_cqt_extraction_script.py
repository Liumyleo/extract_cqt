import numpy as np
import essentia.standard as standard

def essentia_cqt(in_path):
    kwargs = {
        'inputSize': 4096,
        'minFrequency': 32.7,
        'maxFrequency': 4000,
        'binsPerOctave': 12,
        'sampleRate': 22050,
        'rasterize': 'full',
        'phaseMode': 'global',
        'gamma': 0,
        'normalize': 'impulse',
        'window': 'hannnsgcq',
    }
    x = standard.MonoLoader(filename = in_path, sampleRate = 22050)()
    # Remove the last sample to make the signal even
    if len(x) %2 ==1:
        x=x[:-1]

    kwargs['inputSize'] = len(x)
    CQStand = standard.NSGConstantQ(**kwargs)

    constantq, _, _ = CQStand(x)
    constantq = np.abs(constantq)

    #### MEAN SIZE #####
    mean_size = 100
    height, length = constantq.shape
    new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
    for i in range(int(length/mean_size)-1):
        new_cqt[:,i] = constantq[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
    return new_cqt.astype(np.float32)

if __name__=='__main__':
    cqt = essentia_cqt("./002N-hprwiw.mp4")
    print(cqt.shape)

import argparse as ag
import importlib.machinery as im
import os
import numpy as np
import shutil
import soundfile as sf
import tensorflow as tf
import tfmpl
import types

# Dimension
ABS_INT16 = 32767.
BATCH_SIZE = None
MODEL_SIZE = 16  # Matches the model size inside the NNs
MODES = None
WAV_LENGTH = None
Z_LENGTH = 100

# Learning
BETA1 = 0.5
BETA2 = 0.9
LEARN_RATE = None  # 0.0001

# Loss Constants
LAMBDA = None  # 10 for WGAN
LOSS_MAX = 500

# Messages
TRAIN_COMPLETE = False

# MinMax
D_UPDATES_PER_G_UPDATES = 1  # 1 for WGAN
G_UPDATES_PER_D_UPDATES = 1  # 1 for WGAN

# Objects
NETWORKS = None

# Tensor Management
CHECKPOINTS = 5000  # 5000
ITERATIONS = None  # 40000
PROGRAM_MODE = None
OUTPUT_DIR = None
SAMPLE_SAVE_RATE = 1000  # 1000
STEPS = 100  # 100


def main(args):
    """ Runs the relevant command passed through arguments """

    global BATCH_SIZE
    BATCH_SIZE = args.batch

    global ITERATIONS
    ITERATIONS = args.iterations

    global LAMBDA
    LAMBDA = args.lamb

    global MODES
    MODES = len(args.words)

    global WAV_LENGTH
    WAV_LENGTH = args.wave

    global D_UPDATES_PER_G_UPDATES
    global G_UPDATES_PER_D_UPDATES
    D_UPDATES_PER_G_UPDATES = args.D_updates
    G_UPDATES_PER_D_UPDATES = args.G_updates

    global MODEL_SIZE
    MODEL_SIZE = args.modelSize

    global LEARN_RATE
    LEARN_RATE = args.learnRate

    global PROGRAM_MODE
    PROGRAM_MODE = args.mode[0]

    # Parameter Search
    if args.mode[0] == 'search':
        for param in [0.000005, 0.000001]:
            LEARN_RATE = param
            for i in range(0, 10):
                tf.reset_default_graph()
                runName = args.runName[0] + '_param' + \
                    str(param) + '_' + str(i)
                model_dir = _setup(runName, args.model[0])
                print("Experiment:" + runName)
                _train(args.words, runName, model_dir, args.model[0])

    # Train to complete
    elif args.mode[0] == "complete":
        global TRAIN_COMPLETE
        TRAIN_COMPLETE = False
        while not TRAIN_COMPLETE:
            tf.reset_default_graph()
            model_dir = _setup(args.runName[0], args.model[0])
            _train(args.words, args.runName[0], model_dir, args.model[0])

    # Training mode
    elif args.mode[0] == "train":
        model_dir = _setup(args.runName[0], args.model[0])
        _train(args.words, args.runName[0], model_dir, args.model[0])

    # Restart Training
    elif args.mode[0] == "restart":
        _restart(
            args.runName[0],
            args.model[0],
            args.checkpointNum
        )

    # Generator mode
    elif args.mode[0] == "gen":
        _generate(
            args.runName[0],
            args.checkpointNum,
            args.genMode,
            args.model[0],
            args.genLength
        )

    return


def _setup(runName, model):
    model_dir = _modelDirectory(runName, model)
    os.makedirs(model_dir)
    _createGenGraph(model_dir, model)
    return model_dir


def _train(folders, runName, model_dir, model):
    """ Trains the WaveGAN model """

    # Prepare the data
    audio_loader = _loadAudioModule()
    training_data_path = "Data/"
    audio_loader.prepareData(training_data_path, folders)

    # Create generated data
    Z_x, Z_y, Z_yFill = _makeGenerated(True, None)

    # Prepare real data
    X, X_y = audio_loader.loadTrainData()
    epochSize = len(X)

    # TESTING SECTION
    # X = [np.array(X[X_y.index(0)]), np.array(X[X_y.index(1)+1])]
    # X_y = [0, 1]
    # TESTING SECTION

    X = _makeIterators(
        tf.reshape(
            tf.convert_to_tensor(np.vstack(X), dtype=tf.float32),
            [len(X), WAV_LENGTH, 1]
        ),
        X_y,
        len(X),
        WAV_LENGTH
    )

    # Prepare link to the NNs
    global NETWORKS
    NETWORKS = _loadNetworksModule(
        'Networks-' + model + '-' + str(WAV_LENGTH) + '.py',
        'Networks-' + model + '-' + str(WAV_LENGTH) + '.py'
    )

    # Create networks
    if model == 'WGAN':
        with tf.variable_scope('G'):
            G = NETWORKS.generator(Z_x)
        with tf.variable_scope('D'):
            R = NETWORKS.discriminator(X["x"])
        with tf.variable_scope('D', reuse=True):
            F = NETWORKS.discriminator(G)
    elif model == 'CWGAN':
        with tf.variable_scope('G'):
            G = NETWORKS.generator(Z_x, Z_y)
        with tf.variable_scope('D'):
            R = NETWORKS.discriminator(X["x"], X["yFill"])
        with tf.variable_scope('D', reuse=True):
            F = NETWORKS.discriminator(G, Z_yFill)

    # Create variables
    G_variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='G'
    )
    D_variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='D'
    )

    # Build loss
    if model == 'WGAN':
        G_loss, D_loss = _wasser_loss(G, R, F, X)
    elif model == 'CWGAN':
        G_loss, D_loss = _conditioned_wasser_loss(G, R, F, X)

    # Build optimizers
    G_opt = tf.train.AdamOptimizer(
        learning_rate=LEARN_RATE,
        beta1=BETA1,
        beta2=BETA2
    )
    D_opt = tf.train.AdamOptimizer(
        learning_rate=LEARN_RATE,
        beta1=BETA1,
        beta2=BETA2
    )

    # Build training operations
    G_train_op = G_opt.minimize(
        G_loss,
        var_list=G_variables,
        global_step=tf.train.get_or_create_global_step()
    )
    D_train_op = D_opt.minimize(
        D_loss,
        var_list=D_variables
    )

    # Root Mean Square
    Z_rms = tf.sqrt(tf.reduce_mean(tf.square(G[:, :, 0]), axis=1))
    X_rms = tf.sqrt(tf.reduce_mean(tf.square(X["x"][:, :, 0]), axis=1))

    # Plot wave
    X_plotWave = plotWave(X["x"])
    G_plotWave = plotWave(G)

    # Plot spectrogram
    X_plotSpec = plotSpec(X["x"])
    G_plotSpec = plotSpec(G)

    # Summary
    with tf.name_scope('Audio'):
        tf.summary.audio(
            name='X',
            tensor=X["x"],
            sample_rate=WAV_LENGTH,
            max_outputs=6
        )
        tf.summary.audio(
            name='G',
            tensor=G,
            sample_rate=WAV_LENGTH,
            max_outputs=6
        )

    with tf.name_scope('WavePlot'):
        tf.summary.image(
            name='plotWave_X',
            tensor=X_plotWave,
            max_outputs=6
        )
        tf.summary.image(
            name='plotWave_G',
            tensor=G_plotWave,
            max_outputs=6
        )

    with tf.name_scope('plotSpec'):
        tf.summary.image(
            name='plotSpec_X',
            tensor=X_plotSpec,
            max_outputs=6
        )
        tf.summary.image(
            name='plotSpec_G',
            tensor=G_plotSpec,
            max_outputs=6
        )

    with tf.name_scope('Rms'):
        tf.summary.histogram('rms_batch_Z', Z_rms)
        tf.summary.histogram('rms_batch_X', X_rms)
        tf.summary.scalar('rms_X', tf.reduce_mean(X_rms))
        tf.summary.scalar('rms_Z', tf.reduce_mean(Z_rms))

    with tf.name_scope('Loss'):
        tf.summary.scalar('loss_G', G_loss)
        tf.summary.scalar('loss_D', D_loss)

    # Print hyperparameter summary
    with open(model_dir + 'hyperparameters.txt', 'w') as f:
        f.write('Batch Size    : ' + str(BATCH_SIZE) + '\n')
        f.write('Checkpoints   : ' + str(CHECKPOINTS) + '\n')
        f.write('D Updates     : ' + str(D_UPDATES_PER_G_UPDATES) + '\n')
        f.write('Epoch Size    : ' + str(epochSize) + '\n')
        f.write('G Updates     : ' + str(G_UPDATES_PER_D_UPDATES) + '\n')
        f.write('Iterations    : ' + str(ITERATIONS) + '\n')
        f.write('Lambda        : ' + str(LAMBDA) + '\n')
        f.write('Learning Rate : ' + str(LEARN_RATE) + '\n')
        f.write('Model Size    : ' + str(MODEL_SIZE) + '\n')
        f.write('Model Type    : ' + model + '\n')
        f.write('Modes         : ' + str(MODES) + '\n')
        f.write('Wave length   : ' + str(WAV_LENGTH) + '\n')
        f.close

    # Create a session
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_dir,
        config=tf.ConfigProto(log_device_placement=False),
        save_checkpoint_steps=CHECKPOINTS,
        save_summaries_steps=STEPS
    )

    # Run the session
    _runSession(
        sess,
        D_train_op,
        D_loss,
        G_train_op,
        G_loss,
        G,
        model_dir
    )

    return


def _runSession(sess, D_train_op, D_loss, G_train_op, G_loss, G, model_dir):
    """ Runs a session """

    runawayLoss = False
    print("Starting experiment . . .")

    # Update model ITERATIONS number of times
    for iteration in range(1, ITERATIONS + 1):

        # Run Discriminator
        for D_update in range(D_UPDATES_PER_G_UPDATES):
            _, run_D_loss = sess.run([D_train_op, D_loss])
            if abs(run_D_loss) > LOSS_MAX:
                runawayLoss = True
            if runawayLoss:
                break

        # Stop if the Generator loss starts to accelerate
        if runawayLoss:
            print("Ending: D loss = " + str(run_D_loss))
            break

        # Run Generator
        for G_update in range(G_UPDATES_PER_D_UPDATES):
            _, run_G_loss, G_data = sess.run([G_train_op, G_loss, G])
            if abs(run_G_loss) > LOSS_MAX:
                runawayLoss = True
            if runawayLoss:
                break

        # Stop if the Generator loss starts to accelerate
        if runawayLoss:
            print("Ending: G loss = " + str(run_G_loss))
            break

        # Save samples every SAMPLE_SAVE_RATE steps
        if iteration % SAMPLE_SAVE_RATE == 0:
            # fileName = 'Run_' + str(iteration)
            # _saveGenerated(
            #     model_dir + 'Checkpoint_Samples/',
            #     G_data,
            #     fileName
            # )
            print('Completed Iteration: ' + str(iteration))

    sess.close()
    if runawayLoss and not PROGRAM_MODE == "train":
        shutil.rmtree(model_dir)
    elif not runawayLoss:
        global TRAIN_COMPLETE
        TRAIN_COMPLETE = True

    print("Completed experiment.")
    return


def _loadNetworksModule(modName, modPath):
    """ Loads the module containing the relevant networks """

    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    return mod


def _wasser_loss(G, R, F, X):
    """ Calculates the loss """

    # Cost functions
    G_loss = -tf.reduce_mean(F)
    D_loss = tf.reduce_mean(F) - tf.reduce_mean(R)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = G - X["x"]
    interpolates = X["x"] + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
        D_interp = NETWORKS.discriminator(interpolates)

    # Gradient penalty
    gradients = tf.gradients(D_interp, [interpolates], name='grads')[0]
    slopes = tf.sqrt(
        tf.reduce_sum(
            tf.square(gradients),
            reduction_indices=[1, 2]
        )
    )
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

    # Discriminator loss
    D_loss += LAMBDA * gradient_penalty

    # Summaries
    with tf.name_scope('Various'):
        tf.summary.scalar('norm', tf.norm(gradients))
        tf.summary.scalar('grad_penalty', gradient_penalty)

    return G_loss, D_loss


def _conditioned_wasser_loss(G, R, F, X):
    """ Calculates the loss """

    # Cost functions
    G_loss = -tf.reduce_mean(F)
    D_loss = tf.reduce_mean(F) - tf.reduce_mean(R)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1, 1],
        minval=0.,
        maxval=1.
    )
    x_hat = X["x"] * alpha + (1 - alpha) * G
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
        D_interp = NETWORKS.discriminator(x_hat, X["yFill"])

    # Gradient penalty
    gradients = tf.gradients(D_interp, x_hat, name='grads')[0]
    slopes = tf.sqrt(
        tf.reduce_sum(
            tf.square(gradients),
            axis=[1, 2]
        )
    )
    gradient_penalty = LAMBDA * tf.reduce_mean((slopes - 1.) ** 2.)

    # Discriminator loss
    D_loss += gradient_penalty

    # Summaries
    with tf.name_scope('Various'):
        tf.summary.scalar('norm', tf.norm(gradients))
        tf.summary.scalar('grad_penalty', gradient_penalty)

    return G_loss, D_loss


def _modelDirectory(runName, model):
    """ Creates / obtains the name of the model directory """
    directory = 'tmp/' + model + '_' + str(WAV_LENGTH) + '_' + runName + '/'
    return directory


def _makeGenerated(random, genMode):
    """ Makes the tensor generators """

    if random:
        one_hot = tf.random_uniform(
            [BATCH_SIZE, 1, 1],
            0,
            MODES,
            dtype=tf.int32
        )
    else:
        one_hot = tf.fill(
            [BATCH_SIZE, 1, 1],
            genMode
        )

    Z_x = tf.random_uniform(
        [BATCH_SIZE, 1, Z_LENGTH],
        -1.,
        1.,
        dtype=tf.float32
    )
    Z_one_hot = tf.one_hot(
        indices=one_hot[:, 0],
        depth=MODES
    )
    Z_y = tf.multiply(
        x=Z_one_hot,
        y=tf.ones(
            [BATCH_SIZE, MODEL_SIZE, MODES],
            dtype=tf.float32
        )
    )
    Z_yFill = tf.multiply(
        x=Z_one_hot,
        y=tf.ones(
            [BATCH_SIZE, WAV_LENGTH, MODES],
            dtype=tf.float32
        )
    )

    return Z_x, Z_y, Z_yFill


def _makeIterators(data, labels, data_size, data_length):
    """ Creates iterators for the data """

    oneHot = np.zeros((data_size, 1, MODES), dtype=np.float32)
    oneHot[np.arange(data_size), 0, labels] = 1.0
    oneHotFill = oneHot * \
        np.ones((data_size, WAV_LENGTH, MODES), dtype=np.float32)
    oneHot = tf.convert_to_tensor(oneHot, dtype=tf.float32)
    oneHotFill = tf.reshape(
        tensor=tf.cast(oneHotFill, tf.float32),
        shape=[data_size, WAV_LENGTH, MODES]
    )

    dSet = tf.data.Dataset.from_tensor_slices(
        {
            "x": data,
            "y": oneHot,
            "yFill": oneHotFill
        }
    )
    dSet = dSet.shuffle(buffer_size=data_size)
    dSet = dSet.apply(
        tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)
    )
    dSet = dSet.repeat()

    iterator = dSet.make_one_shot_iterator()
    iterator = iterator.get_next()

    return iterator


def _loadAudioModule():
    """ Loads the audio module & returns it as objects """
    audio_loader = _loadNetworksModule(
        'audioDataLoader.py',
        'audioDataLoader.py'
    )
    return audio_loader


def _createConfigFile():
    """ Creates a config file for the training session """
    return


def _createGenGraph(model_dir, model):
    """ Creates a copy of the generator graph """

    # Prepare link to the NNs
    global NETWORKS
    NETWORKS = _loadNetworksModule(
        'Networks-' + model + '-' + str(WAV_LENGTH) + '.py',
        'Networks-' + model + '-' + str(WAV_LENGTH) + '.py'
    )

    # Create directory
    graphDir = os.path.join(model_dir + 'Generator/')
    if not os.path.isdir(graphDir):
        os.makedirs(graphDir)

    # Create graph
    Z_Input = tf.placeholder(tf.float32, [None, 1, Z_LENGTH], name='Z_Input')
    Z_Labels = tf.placeholder(
        tf.float32, [None, MODEL_SIZE, MODES], name='Z_Labels')

    if model == 'WGAN':
        with tf.variable_scope('G'):
            G = NETWORKS.generator(Z_Input)
        G = tf.identity(G, name='Generator')
    elif model == 'CWGAN':
        with tf.variable_scope('G'):
            G = NETWORKS.generator(Z_Input, Z_Labels)
        G = tf.identity(G, name='Generator')

    # Save graph
    G_variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='G'
    )
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(G_variables + [global_step])

    # Export All
    tf.train.write_graph(
        tf.get_default_graph(),
        graphDir,
        'generator.pbtxt'
    )
    tf.train.export_meta_graph(
        filename=os.path.join(graphDir, 'generator.meta'),
        clear_devices=True,
        saver_def=saver.as_saver_def()
    )
    tf.reset_default_graph()

    return


def _restart(runName, model, ckptNum):
    """ Restarts the training of the model """

    model_dir = _modelDirectory(runName, model)

    tf.reset_default_graph()
    sess = tf.Session()
    ckpt = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.import_meta_graph(
        model_dir + 'model.ckpt-' + str(ckptNum) + '.meta'
    )
    saver.restore(sess, ckpt)
    graph = tf.get_default_graph()

    _runSession(
        sess,
        graph.get_operation_by_name('Adam_1'),
        graph.get_tensor_by_name('add_1:0'),
        graph.get_operation_by_name('Adam'),
        graph.get_tensor_by_name('Neg:0'),
        graph.get_collection('G'),
        model_dir
    )

    sess.close()

    return


def _generate(runName, checkpointNum, genMode, model, genLength):
    """ Generates samples from the generator """

    # Load the graph
    model_dir = _modelDirectory(runName, model)
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    tf.train.import_meta_graph(
        model_dir + 'Generator/generator.meta'
    ).restore(
        sess,
        model_dir + 'model.ckpt-' + str(checkpointNum)
    )

    # Generate sounds
    Z = np.random.uniform(-1., 1., [genLength, 1, Z_LENGTH])

    # Get tensors
    Z_input = graph.get_tensor_by_name('Z_Input:0')
    G = graph.get_tensor_by_name('Generator:0')

    # Enter into graph
    if model == 'WGAN':
        samples = sess.run(G, {Z_input: Z})
    elif model == 'CWGAN':
        Z_y = np.zeros(
            shape=(genLength, MODEL_SIZE, MODES),
            dtype=np.int16
        )
        Z_y[:, :, genMode] = 1
        Z_labels = graph.get_tensor_by_name('Z_Labels:0')
        samples = sess.run(G, {Z_input: Z, Z_labels: Z_y})

    # Create the output path
    path = os.path.abspath(
        os.path.join(
            os.path.dirname((__file__)),
            os.pardir,
            'Generated/',
            model + '_' + str(WAV_LENGTH) + '/',
            'ModelRun_' + str(runName)
        )
    )

    if model == 'WGAN':
        fileName = 'Random'
    elif model == 'CWGAN':
        fileName = 'Mode_' + str(genMode)

    # Write samples to file
    _saveGenerated(path, samples, fileName)

    return


def _saveGenerated(path, samples, fileName):
    """ Saves the generated samples to folder as .wav """

    if not os.path.exists(path):
        os.makedirs(path)

    # Save the samples
    i = 0
    for sample in samples:
        i = i + 1
        sf.write(
            file=path + '/' + fileName + '_' + str(i) + '.wav',
            data=sample,
            samplerate=WAV_LENGTH,
            subtype='PCM_16'
        )

    return


@tfmpl.figure_tensor
def plotWave(audioTensor):
    """ Plots audio to a wave graph """
    figs = tfmpl.create_figures(BATCH_SIZE)
    for i, f in enumerate(figs):
        sp = f.add_subplot(111)
        sp.plot(audioTensor[i, :, :])

    return figs


@tfmpl.figure_tensor
def plotSpec(audioTensor):
    """ Plots audio to a wave graph """
    figs = tfmpl.create_figures(BATCH_SIZE)
    for i, f in enumerate(figs):
        sp = f.add_subplot(111)
        dim = audioTensor[i, :, :]
        dim = np.squeeze(dim)
        sp.specgram(
            x=dim,
            NFFT=256,
            Fs=2
        )

    return figs


if __name__ == "__main__":
    parser = ag.ArgumentParser()
    parser.add_argument(
        '-mode',
        nargs=1,
        type=str,
        default='train',
        help="How you wish to use the model."
    )
    parser.add_argument(
        '-model',
        nargs=1,
        type=str,
        help="Which generator type do you want to use?"
    )
    parser.add_argument(
        '-wave',
        type=int,
        default=1024,
        help="The wave length of files for this experiment"
    )
    parser.add_argument(
        '-runName',
        nargs=1,
        type=str,
        help="A name for this run of the experiment."
    )
    parser.add_argument(
        '-checkpointNum',
        type=int,
        help="The checkpoint number you wish to examine."
    )
    parser.add_argument(
        '-genMode',
        type=int,
        default=0,
        help="The number of the mode to be generated."
    )
    parser.add_argument(
        '-genLength',
        type=int,
        default=100,
        help="The count of samples you want generated."
    )
    parser.add_argument(
        '-lamb',
        type=int,
        default=10,
        help="The lambda to be applied to Wasserstein Loss."
    )
    parser.add_argument(
        '-batch',
        type=int,
        default=64,
        help="The batch size."
    )
    parser.add_argument(
        '-iterations',
        type=int,
        default=1000,
        help="The number of times you want the model to run."
    )
    parser.add_argument(
        '-D_updates',
        type=int,
        default=1,
        help="The number of discriminator updates to generator."
    )
    parser.add_argument(
        '-G_updates',
        type=int,
        default=1,
        help="The number of generator updates to discriminator."
    )
    parser.add_argument(
        '-modelSize',
        type=int,
        default=16,
        help="The depth of the model being used."
    )
    parser.add_argument(
        '-learnRate',
        type=float,
        default=0.0001,
        help="The learning rate used."
    )
    parser.add_argument(
        '-words',
        nargs='*',
        type=str,
        default=['zero', 'one'],
        help="The words for sounds you want to train with."
    )
    main(parser.parse_args())

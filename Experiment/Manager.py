import argparse as ag
import importlib.machinery as im
import os
import numpy as np
import soundfile as sf
import tensorflow as tf
import types

ABS_INT16 = 32767.
BATCH_SIZE = 64
BETA1 = 0.5
BETA2 = 0.9
CHECKPOINTS = 10000  # 10000
D_UPDATES_PER_G_UPDATES = 1
GEN_LENGTH = 100
ITERATIONS = 200000  # 200000
LAMBDA = 10
LEARN_RATE = 0.0001
LOSS_MAX = 100
MODES = 2
NETWORKS = None
OUTPUT_DIR = None
SAMPLE_SAVE_RATE = 10000
STEPS = 100
WAV_LENGTH = None
Z_LENGTH = 100


def main(args):
    """ Runs the relevant command passed through arguments """

    global WAV_LENGTH
    WAV_LENGTH = args.wave[0]

    # Training mode
    if args.mode[0] == "train":
        # Train model
        model_dir = _modelDirectory(args.runName[0], args.model[0])
        _createGenGraph(model_dir, args.model[0])
        _train(args.words, args.runName[0], model_dir, args.model[0])

    # Generator mode
    elif args.mode[0] == "gen":
        _generate(
            args.runName[0],
            args.checkpointNum[0],
            args.genMode,
            args.model[0]
        )

    return


def _train(folders, runName, model_dir, model):
    """ Trains the WaveGAN model """

    # Prepare the data
    audio_loader = _loadAudioModule()
    training_data_path = "Data/"
    audio_loader.prepareData(training_data_path, folders)
    gen_path = model_dir + 'Checkpoint_Samples/'

    # Create generated data
    Z = _makeIterators(
        np.random.uniform(-1., 1., [BATCH_SIZE, 1, Z_LENGTH]),
        np.random.randint(0, MODES, (BATCH_SIZE, 1)),
        BATCH_SIZE,
        Z_LENGTH
    )

    # Prepare real data
    X, X_y = audio_loader.loadTrainData()
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
            G = NETWORKS.generator(Z["x"])
        with tf.variable_scope('D'):
            R = NETWORKS.discriminator(X["x"])
        with tf.variable_scope('D', reuse=True):
            F = NETWORKS.discriminator(G)
    elif model == 'CWGAN':
        with tf.variable_scope('G'):
            G = NETWORKS.generator(Z["x"], Z["y"])
        with tf.variable_scope('D'):
            R, R_logits = NETWORKS.discriminator(X["x"], X["yFill"])
        with tf.variable_scope('D', reuse=True):
            F, F_logits = NETWORKS.discriminator(G, Z["yFill"])

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
        G_loss, D_loss = _base_loss(G, R, F, X["x"], Z["x"])
    elif model == 'CWGAN':
        G_loss, D_loss = _conditioned_loss(R_logits, F_logits)

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

    # Summary
    tf.summary.audio('X', X["x"], WAV_LENGTH)
    tf.summary.audio('G', G, WAV_LENGTH)
    tf.summary.histogram('Z_rms', Z_rms)
    tf.summary.histogram('X_rms', X_rms)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Run session
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_dir,
        config=tf.ConfigProto(log_device_placement=False),
        save_checkpoint_steps=CHECKPOINTS,
        save_summaries_steps=STEPS
    )

    print("Starting experiment . . .")
    for iteration in range(1, ITERATIONS + 1):

        # Run Discriminator
        for D_update in range(D_UPDATES_PER_G_UPDATES):
            _, run_D_loss = sess.run([D_train_op, D_loss])
            if abs(run_D_loss) > LOSS_MAX:
                print("Ending: D loss = " + str(run_D_loss))
                break

        # Run Generator
        _, run_G_loss, G_data = sess.run([G_train_op, G_loss, G])
        if abs(run_G_loss) > LOSS_MAX:
            print("Ending: G Loss = " + str(run_G_loss))
            break

        if iteration % SAMPLE_SAVE_RATE == 0:
            fileName = 'Run_' + str(iteration)
            _saveGenerated(gen_path, G_data, fileName)
            print('Completed Iteration: ' + str(iteration))

    return


def _loadNetworksModule(modName, modPath):
    """ Loads the module containing the relevant networks """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def _base_loss(G, R, F, X, Z):
    """ Calculates the loss """
    G_loss = -tf.reduce_mean(F)
    D_loss = tf.reduce_mean(F) - tf.reduce_mean(R)
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = G - X
    interpolates = X + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
        D_interp = NETWORKS.discriminator(interpolates)
    gradients = tf.gradients(D_interp, [interpolates], name='grads')[0]
    slopes = tf.sqrt(
        tf.reduce_sum(
            tf.square(gradients),
            reduction_indices=[1, 2]
        )
    )
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
    tf.summary.scalar('norm', tf.norm(gradients))
    tf.summary.scalar('grad_penalty', gradient_penalty)
    return G_loss, D_loss


def _conditioned_loss(R_logits, F_logits):
    """ Calculates the loss """
    G_loss = -tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=F_logits,
            labels=tf.ones([BATCH_SIZE, MODES])
        )
    )
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=R_logits,
            labels=tf.ones([BATCH_SIZE, MODES])
        )
    )
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=F_logits,
            labels=tf.zeros([BATCH_SIZE, MODES])
        )
    )
    D_loss = D_loss_real + D_loss_fake
    return G_loss, D_loss


def _modelDirectory(runName, model):
    """ Creates / obtains the name of the model directory """
    directory = 'tmp/' + model + '_' + str(WAV_LENGTH) + '_' + runName + '/'
    return directory


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
    Z_Labels = tf.placeholder(tf.float32, [None, 1, MODES], name='Z_Labels')

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


def _generate(runName, checkpointNum, genMode, model):
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
    Z = np.random.uniform(-1., 1., [GEN_LENGTH, 1, Z_LENGTH])

    # Get tensors
    Z_input = graph.get_tensor_by_name('Z_Input:0')
    G = graph.get_tensor_by_name('Generator:0')

    # Enter into graph
    if model == 'WGAN':
        samples = sess.run(G, {Z_input: Z})
    elif model == 'CWGAN':
        # Prepare labels
        oneHot = np.zeros((GEN_LENGTH, 1, MODES), dtype=np.float32)
        oneHot[np.arange(GEN_LENGTH), 0, genMode] = 1.0
        Z_labels = graph.get_tensor_by_name('Z_Labels:0')
        # Computex
        samples = sess.run(G, {Z_input: Z, Z_labels: oneHot})

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
        nargs=1,
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
        nargs=1,
        type=int,
        help="The checkpoint number you wish to examine."
    )
    parser.add_argument(
        '-genMode',
        nargs=1,
        type=int,
        default=0,
        help="The number of the mode to be generated."
    )
    parser.add_argument(
        '-words',
        nargs='*',
        type=str,
        default=['zero', 'one'],
        help="The words for sounds you want to train with."
    )
    main(parser.parse_args())

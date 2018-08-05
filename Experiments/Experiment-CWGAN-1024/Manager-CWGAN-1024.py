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
D_UPDATES_PER_G_UPDATES = 1
GEN_LENGTH = 100
LAMBDA = 10
LEARN_RATE = 0.0001
LOSS_MAX = 100
MODES = 2
NETWORKS = None
OUTPUT_DIR = None
RUNS = 10
WAV_LENGTH = 1024
Z_LENGTH = 100


def main(args):
    """ Runs the relevant command passed through arguments """

    # Training mode
    if args.mode[0] == "train":
        # Train model
        model_dir = _modelDirectory(args.runName[0])
        _createGenGraph(model_dir)
        _train(args.words, args.runName[0], model_dir)

    # Generator mode
    elif args.mode[0] == "gen":
        _generate(args.runName[0], args.checkpointNum)

    return


def _train(folders, runName, model_dir):
    """ Trains the WaveGAN model """

    # Prepare the data
    audio_loader = _loadAudioModule()
    training_data_path = "Data/"
    audio_loader.prepareData(training_data_path, folders)

    # Create generated data
    Z = tf.random_uniform([BATCH_SIZE, Z_LENGTH], -1., 1., dtype=tf.float32)
    Z_y = np.random.randint(0, MODES-1, (BATCH_SIZE, 1))
    zOneHotLabels = np.zeros((BATCH_SIZE, max(Z_y)))[
        np.arrange(BATCH_SIZE), Z_y] = 1
    Z_y = zOneHotLabels * np.ones([BATCH_SIZE, MODES])

    # Prepare real data
    X, X_labels = audio_loader.loadAllData()
    X_length = len(X)
    X = np.vstack(X)
    X = tf.reshape(
        tensor=tf.cast(X, tf.float32),
        shape=[X_length, WAV_LENGTH, 1]
    )
    X = tf.data.Dataset.from_tensor_slices(X)
    X = X.shuffle(buffer_size=X_length)
    X = X.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    X = X.repeat()
    X = X.make_one_shot_iterator()
    X = X.get_next()

    # Prepare real data labels
    X_y = tf.placeholder(
        tf.float32,
        shape=[X_length, WAV_LENGTH, MODES]
    )
    oneHotLabels = np.zeros((X_length, max(X_labels)))[
        np.arrange(X_length), X_labels] = 1
    X_y = oneHotLabels * np.ones([X_length, WAV_LENGTH, MODES])
    X_y = tf.data.Dataset.from_tensor_slices(X_y)
    X_y = X_y.shuffle(buffer_size=X_length)
    X_y = X_y.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    X_y = X_y.repeat()
    X_y = X_y.make_one_shot_iterator()
    X_y = X_y.get_next()

    # Prepare link to the NNs
    global NETWORKS
    NETWORKS = _loadNetworksModule(
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py',
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py'
    )

    # Create networks
    with tf.variable_scope('G'):
        G = NETWORKS.generator(Z, Z_y)
    with tf.variable_scope('D'):
        R, R_logits = NETWORKS.discriminator(X, X_y)
    with tf.variable_scope('D', reuse=True):
        F, F_logits = NETWORKS.discriminator(G, X_y)

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
    G_loss, D_loss = _loss(R_logits, F_logits)

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
    X_rms = tf.sqrt(tf.reduce_mean(tf.square(X[:, :, 0]), axis=1))

    # Summary
    tf.summary.audio('X', X, WAV_LENGTH)
    tf.summary.audio('G', G, WAV_LENGTH)
    tf.summary.histogram('Z_rms', Z_rms)
    tf.summary.histogram('X_rms', X_rms)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Run session
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_dir,
        config=tf.ConfigProto(log_device_placement=False),
        save_checkpoint_steps=10000,
        save_summaries_steps=100
    )

    for epoch in range(RUNS):

        if epoch % 1000 == 0:
            print('Completed Run Number: ' + str(epoch))

        # Run Discriminator
        for D_update in range(D_UPDATES_PER_G_UPDATES):
            _, run_D_loss = sess.run([D_train_op, D_loss])
            if abs(run_D_loss) > LOSS_MAX:
                print("Ending: D loss = " + str(run_D_loss))
                break

        # Run Generator
        _, run_G_loss = sess.run([G_train_op, G_loss])
        if abs(run_G_loss) > LOSS_MAX:
            print("Ending: G Loss = " + str(run_G_loss))
            break

    return


def _loadNetworksModule(modName, modPath):
    """ Loads the module containing the relevant networks """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def _loss(R_logits, F_logits):
    """ Calculates the loss """
    G_loss = -tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=F_logits,
            labels=tf.ones(BATCH_SIZE, 1, 1)
        )
    )
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=R_logits,
            labels=tf.ones(BATCH_SIZE, 1, 1)
        )
    )
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=F_logits,
            labels=tf.zeros(BATCH_SIZE, 1, 1)
        )
    )
    D_loss = D_loss_real + D_loss_fake
    return G_loss, D_loss


def _modelDirectory(runName):
    """ Creates / obtains the name of the model directory """
    return 'tmp/testWaveGAN_' + str(WAV_LENGTH) + '_' + runName + '/'


def _loadAudioModule():
    """ Loads the audio module & returns it as objects """
    audio_loader = _loadNetworksModule(
        'audioDataLoader.py',
        'audioDataLoader.py'
    )
    return audio_loader


def _createGenGraph(model_dir):
    """ Creates a copy of the generator graph """

    # Prepare link to the NNs
    global NETWORKS
    NETWORKS = _loadNetworksModule(
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py',
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py'
    )

    # Create directory
    graphDir = os.path.join(model_dir + 'Generator/')
    if not os.path.isdir(graphDir):
        os.makedirs(graphDir)

    # Create graph
    Z_Input = tf.placeholder(tf.float32, [None, Z_LENGTH], name='Z_Input')
    with tf.variable_scope('G'):
        G = NETWORKS.generator(Z_Input)
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


def _generate(runName, checkpointNum):
    """ Generates samples from the generator """

    # Load the graph
    model_dir = _modelDirectory(runName)
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
    Z = np.random.uniform(-1., 1., [GEN_LENGTH, Z_LENGTH])
    Z_input = graph.get_tensor_by_name('Z_Input:0')
    G = graph.get_tensor_by_name('Generator:0')
    samples = sess.run(G, {Z_input: Z})

    # Write samples to file
    _saveGenerated(samples, runName)

    return


def _saveGenerated(samples, runName):
    """ Saves the generated samples to folder as .wav """

    # Create the output path
    path = os.path.abspath(
        os.path.join(
            os.path.dirname((__file__)),
            os.pardir,
            os.pardir,
            'Generated/',
            str(WAV_LENGTH) + '/',
            'ModelRun_' + str(runName)
        )
    )
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the samples
    i = 0
    for sample in samples:
        i = i + 1
        sf.write(
            file=path + '/' + 'Sample_' + str(i) + '.wav',
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
        '-runName',
        nargs=1,
        type=str,
        help="A name for this run of the experiment."
    )
    parser.add_argument(
        '-checkpointNum',
        nargs='?',
        type=int,
        help="The checkpoint number you wish to examine."
    )
    parser.add_argument(
        '-words',
        nargs='*',
        type=str,
        default=['zero'],
        help="The words for sounds you want to train with."
    )
    main(parser.parse_args())

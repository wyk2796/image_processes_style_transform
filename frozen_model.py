import os
import tensorflow as tf
import params as P
import util

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_path, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    # We precise the file fullname of our freezed graph
    # absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = output_path + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.keras.backend.set_learning_phase(0)
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph

        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        # We restore the weights
        # graphdef_inf = tf.graph_util.remove_training_nodes(input_graph_def)
        saver.restore(sess, input_checkpoint)

        # fix batch norm nodes
        #
        # for node in graphdef_inf.node:
        #     print('name:', node.name)
        #     if node.op == 'RefSwitch':
        #         print(node.op, 'Switch')
        #         node.op = 'Switch'
        #         for index in range(len(node.input)):
        #             if 'moving_' in node.input[index]:
        #                 print("remove_move")
        #                 node.input[index] = node.input[index] + '/read'
        #     elif node.op == 'AssignSub':
        #         node.op = 'Sub'
        #         if 'use_locking' in node.attr: del node.attr['use_locking']

        # util.print_tensor_name()
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def convert_to_tfjs(frozen_path, output_path, output_node_name, model_type):
    util.save_as_tfjs_model(frozen_path, output_node_name, output_path, model_type)


if __name__ == '__main__':
    style_name = 'la_muse'
    save_path = P.saved_model_dir + style_name + '/'
    frozen_model_path = P.frozen_model_dir + style_name + '\\'
    tfjs_model_path = P.tfjs_saved_dir + style_name + '/'
    output_node = ','.join(util.load_tensor_name_from_file(P.st_model_tensor_name_path))
    print(output_node)
    freeze_graph(save_path, frozen_model_path, output_node)
    # convert_to_tfjs(frozen_model_path + 'frozen_model.pb',
    #                 tfjs_model_path, ','.join(util.load_tensor_name_from_file(P.st_model_tensor_name_path)), model_type='frozen_model')

# tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='lambda_8/mul:0' --saved_model_tags=serve frozen_model.pb la_muse
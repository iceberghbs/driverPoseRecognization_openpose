import tensorflow
 
tensorflow.compat.v1.reset_default_graph()
tensorflow.compat.v1.keras.backend.set_learning_phase(0)  # 调用模型前一定要执行该命令
tensorflow.compat.v1.disable_v2_behavior()  # 禁止tensorflow2.0的行为

# 加载hdf5模型 输入模型编号 生成相应的模型的pb文件
model_index = 19  # 模型编号########################## 改这里 ################# 
model_name = 'model' + str(model_index)
hdf5_pb_model = tensorflow.compat.v1.keras.models.load_model('./recognize_net/' + model_name + '.h5')
 
 
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        #         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        #         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names", output_names)
        input_graph_def = graph.as_graph_def()
        #         for node in input_graph_def.node:
        #             print('node:', node.name)
        print("len node1", len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tensorflow.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                     output_names)
 
        outgraph = tensorflow.compat.v1.graph_util.remove_training_nodes(frozen_graph)  # 云掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1", len(outgraph.node))
        return outgraph
 
 
output_folder2 = 'recognize_net'
 
frozen_graph = freeze_session(tensorflow.compat.v1.keras.backend.get_session(),
                              output_names=[out.op.name for out in hdf5_pb_model.outputs])
tensorflow.compat.v1.train.write_graph(frozen_graph, output_folder2, model_name + ".pb", as_text=False)
# import migraphx and numpy
import migraphx
import numpy as np
# import and parse inception model
model = migraphx.parse_onnx("inceptioni1.onnx")
# compile model for the GPU target
model.compile(migraphx.get_target("gpu"))
# optionally print compiled model
model.print()
# create random input image
input_image = np.random.rand(1, 3, 299, 299).astype('float32')
# feed image to model, 'x.1` is the input param name
results = model.run({'x.1': input_image})
# get the results back
result_np = np.array(results[0])
# print the inferred class of the input image
print(np.argmax(result_np))

import predictions as p

onnx_model = '2_Predictions/Model/weights/best.onnx'
test_img = '2_Predictions/00081.jpg'
data_yaml = '2_Predictions/data.yaml'
opt = p.parse_opt(onnx_model, test_img, data_yaml)
p.main(opt)
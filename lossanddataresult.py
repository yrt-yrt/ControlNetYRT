import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例的numpy数组
loss = np.array([0.8495669364929199, 0.8499431610107422, 0.850081205368042, 0.8505493998527527, 0.8511164784431458, 0.8517830967903137, 0.8523537516593933, 0.8528748750686646, 0.8533952832221985, 0.8548036217689514, 0.8550266027450562, 0.8552213907241821, 0.8551442623138428, 0.8552294969558716, 0.8551552891731262, 0.8551148772239685, 0.8549279570579529, 0.854960024356842, 0.854390561580658, 0.8532575964927673, 0.8528978824615479, 0.8525955677032471, 0.8524064421653748, 0.852057695388794, 0.8514969944953918, 0.8510312438011169, 0.85035640001297, 0.8500508069992065, 0.8498373031616211, 0.8494897484779358, 0.8488470315933228, 0.8483119010925293, 0.8477361798286438, 0.8470869064331055, 0.8466013669967651, 0.8445838093757629, 0.8447259068489075, 0.8439003825187683, 0.8429896235466003, 0.8430762887001038, 0.8422922492027283, 0.8430509567260742, 0.8418914079666138, 0.8409119248390198, 0.8406083583831787, 0.8390902280807495, 0.8394945859909058, 0.8405174016952515, 0.8409011363983154, 0.8401774168014526, 0.838182270526886, 0.8394922614097595, 0.8381232023239136, 0.8357980251312256, 0.834087610244751, 0.8312272429466248, 0.8289843797683716, 0.827585756778717, 0.8262692093849182, 0.8251515626907349, 0.8222483396530151, 0.820770263671875, 0.8194212317466736, 0.8170638680458069, 0.8156120181083679, 0.8150956630706787, 0.8163678646087646, 0.8157867193222046, 0.8131428360939026, 0.8094096183776855, 0.8062910437583923, 0.8039264678955078, 0.8001012802124023, 0.7966696619987488, 0.7898791432380676, 0.7880653738975525, 0.7836295962333679, 0.780201256275177, 0.7789034247398376, 0.778838038444519, 0.7755130529403687, 0.7722874283790588, 0.7710958123207092, 0.7681559324264526, 0.7665454149246216, 0.7658877372741699, 0.7648278474807739, 0.7620510458946228, 0.7596644163131714, 0.7571040391921997, 0.7554827928543091, 0.7534831166267395, 0.7510063052177429, 0.7522474527359009, 0.7517687678337097, 0.7488166689872742, 0.7471278309822083, 0.7454057931900024, 0.741797149181366, 0.7418164610862732, 0.7383028268814087, 0.7345959544181824, 0.7311193943023682, 0.7273188829421997, 0.7231383919715881, 0.7158816456794739, 0.7107346653938293, 0.7076215744018555, 0.7045459747314453, 0.7008509635925293, 0.697571337223053, 0.6946389675140381, 0.6907166838645935, 0.6882947683334351, 0.6847638487815857, 0.68063884973526, 0.6758347749710083, 0.6715766191482544, 0.6681818962097168, 0.6629837155342102, 0.6578393578529358, 0.6579365134239197, 0.6545063853263855, 0.6549326777458191, 0.6548498272895813, 0.6534449458122253, 0.6502689719200134, 0.648941695690155, 0.6464952230453491, 0.643449068069458, 0.639568030834198, 0.6372000575065613, 0.6348957419395447, 0.6330368518829346, 0.6310312151908875, 0.6281237602233887, 0.6266739368438721, 0.6243448257446289, 0.6206494569778442, 0.6134397983551025, 0.6106722354888916, 0.6094199419021606, 0.6032260656356812, 0.5993202924728394, 0.5964901447296143, 0.593425452709198, 0.5919505953788757, 0.5896008014678955, 0.5856658816337585, 0.5836114883422852, 0.5795383453369141, 0.5778489708900452, 0.5695235133171082, 0.5672361850738525, 0.5664108991622925, 0.5610860586166382, 0.558691680431366, 0.5570457577705383, 0.5558751821517944, 0.554128110408783, 0.5525671243667603, 0.5507858395576477, 0.5477136969566345, 0.5456962585449219, 0.5447707772254944, 0.5409441590309143, 0.5406285524368286, 0.5404406189918518, 0.5404226779937744, 0.5402673482894897, 0.5394673943519592, 0.537604570388794, 0.5341981649398804, 0.5316329598426819, 0.5292468070983887, 0.5286381244659424, 0.5246520638465881, 0.5243779420852661, 0.5226118564605713, 0.5207516551017761, 0.5162672400474548, 0.5150153040885925, 0.513271689414978, 0.5128523707389832, 0.511320173740387, 0.511836588382721, 0.5117254257202148, 0.5105839967727661, 0.5110319256782532, 0.5117312669754028, 0.5132732391357422, 0.5144182443618774, 0.5156249403953552, 0.5148231387138367, 0.5144293308258057, 0.5167072415351868, 0.5206613540649414, 0.5215641856193542, 0.5230812430381775, 0.5235152244567871, 0.5244737863540649, 0.525877833366394, 0.5269932150840759, 0.5286096930503845, 0.5303067564964294, 0.5328502655029297, 0.5343916416168213, 0.5360326766967773, 0.5371938347816467, 0.5386660695075989, 0.5391180515289307, 0.5456178784370422, 0.5467067360877991, 0.5486650466918945, 0.5508978366851807, 0.5544992089271545, 0.5611532330513, 0.5659894943237305, 0.5684210658073425, 0.5691072344779968, 0.5699699521064758, 0.5721139907836914, 0.5731112360954285, 0.5752128958702087, 0.5789397954940796, 0.5819161534309387, 0.5844976902008057, 0.5891449451446533, 0.5931569337844849, 0.59842449426651, 0.6032206416130066, 0.6102999448776245, 0.6204798221588135, 0.6259989738464355, 0.6278581619262695, 0.6322909593582153, 0.6352135539054871, 0.6380845308303833, 0.6409021615982056, 0.6446934938430786, 0.6481557488441467, 0.6512852907180786, 0.6557644605636597, 0.6591657400131226, 0.6615663766860962, 0.6636877059936523, 0.6646319627761841, 0.6662330627441406, 0.6719651818275452, 0.6760982871055603, 0.6809646487236023, 0.686155378818512, 0.697346031665802, 0.7074460983276367, 0.7163825631141663, 0.7256050109863281, 0.7395842671394348, 0.7510057687759399, 0.7548103332519531, 0.7598992586135864, 0.7720616459846497, 0.7772644758224487, 0.7804097533226013, 0.787145733833313, 0.7876768112182617, 0.7899041771888733, 0.7856791615486145, 0.785387396812439, 0.7950382828712463, 0.7953202724456787, 0.7939228415489197, 0.8033809661865234, 0.8150562644004822, 0.8216183185577393, 0.8255781531333923, 0.8306492567062378, 0.834074079990387, 0.8393996953964233, 0.8518603444099426, 0.8697688579559326, 0.8835064768791199, 0.8948293924331665, 0.9118466973304749, 0.914393961429596, 0.9333088397979736, 0.9513716697692871, 0.963043212890625, 0.9812770485877991, 0.996918797492981, 1.0001184940338135, 1.0064843893051147, 1.0097808837890625, 1.0077311992645264, 1.0159351825714111, 1.0377715826034546, 1.0357531309127808, 1.0337313413619995, 1.0334302186965942, 1.0342755317687988, 1.034319281578064, 1.0308202505111694])

control = np.array([0.9900, 0.9800, 0.9701, 0.9601, 0.9503, 0.9417, 0.9349, 0.9299, 0.9264, 0.9242, 0.9228, 0.9221, 0.9220, 0.9225, 0.9235, 0.9249, 0.9266, 0.9286, 0.9309, 0.9337, 0.9372, 0.9415, 0.9471, 0.9541, 0.9624, 0.9717, 0.9817, 0.9922, 1.0031, 1.0143, 1.0257, 1.0377, 1.0499, 1.0624, 1.0751, 1.0881, 1.1014, 1.1149, 1.1288, 1.1429, 1.1572, 1.1717, 1.1864, 1.2012, 1.2163, 1.2315, 1.2469, 1.2625, 1.2783, 1.2942, 1.3104, 1.3267, 1.3432, 1.3598, 1.3766, 1.3936, 1.4107, 1.4279, 1.4452, 1.4626, 1.48, 1.4975, 1.515, 1.5325, 1.55, 1.5675, 1.585, 1.6026, 1.6202, 1.6379, 1.6556, 1.6734, 1.6912, 1.7091, 1.727, 1.745, 1.7631, 1.7813, 1.7994, 1.8177, 1.8359, 1.8542, 1.8724, 1.8907, 1.909, 1.9272, 1.9454, 1.9635, 1.9817, 1.9998, 2.0179, 2.036, 2.0542, 2.0723, 2.0905, 2.1088, 2.1271, 2.1454, 2.1638, 2.1824, 2.201, 2.2197, 2.2384, 2.2573, 2.2762, 2.2952, 2.3142, 2.3333, 2.3524, 2.3715, 2.3907, 2.4099, 2.4291, 2.4483, 2.4675, 2.4867, 2.5059, 2.5251, 2.5443, 2.5635, 2.5827, 2.6019, 2.6211, 2.6403, 2.6594, 2.6786, 2.6978, 2.717, 2.7361, 2.7553, 2.7744, 2.7936, 2.8127, 2.8319, 2.8511, 2.8702, 2.8894, 2.9086, 2.9277, 2.9469, 2.9661, 2.9853, 3.0045, 3.0237, 3.043, 3.0622, 3.0814, 3.1006, 3.1198, 3.139, 3.1581, 3.1772, 3.1962, 3.2153, 3.2342, 3.2531, 3.272, 3.2908, 3.3096, 3.3282, 3.3469, 3.3655, 3.384, 3.4025, 3.4209, 3.4393, 3.4576, 3.4759, 3.4941, 3.5123, 3.5304, 3.5485, 3.5665, 3.5845, 3.6024, 3.6203, 3.6382, 3.656, 3.6737, 3.6914, 3.709, 3.7266, 3.7442, 3.7617, 3.7792, 3.7966, 3.814, 3.8313, 3.8486, 3.8659, 3.8831, 3.9002, 3.9173, 3.9344, 3.9515, 3.9685, 3.9854, 4.0023, 4.0192, 4.036, 4.0528, 4.0696, 4.0863, 4.1029, 4.1196, 4.1362, 4.1527, 4.1692, 4.1857, 4.2022, 4.2186, 4.235, 4.2513, 4.2676, 4.2839, 4.3002, 4.3164, 4.3326, 4.3488, 4.3649, 4.3811, 4.3972, 4.4133, 4.4293, 4.4454, 4.4615, 4.4775, 4.4935, 4.5096, 4.5256, 4.5416, 4.5576, 4.5736, 4.5896, 4.6056, 4.6215, 4.6375, 4.6534, 4.6694, 4.6853, 4.7012, 4.7171, 4.733, 4.7489, 4.7648, 4.7807, 4.7966, 4.8124, 4.8284, 4.8443, 4.8602, 4.8762, 4.8922, 4.9082, 4.9243, 4.9404, 4.9566, 4.9728, 4.9891, 5.0055, 5.0219, 5.0383, 5.0548, 5.0714, 5.0879, 5.1045, 5.1212, 5.1378, 5.1545, 5.1711, 5.1878, 5.2045, 5.2212, 5.2379, 5.2546, 5.2713, 5.2881, 5.3049, 5.3216, 5.3384, 5.3553, 5.3721, 5.3889, 5.4058, 5.4227, 5.4396, 5.4565, 5.4735, 5.4904, 5.5074, 5.5244, 5.5415, 5.5585, 5.5756, 5.5927, 5.6098, 5.627, 5.6443, 5.6615, 5.6788, 5.6962])

grad = np.array([0.1392, 0.1643, 0.2476, 0.2447, 0.1539, 0.0249, -0.0513, -0.0716, -0.0772, -0.0809, -0.0718, -0.0619, -0.0584, -0.0585, -0.0409, -0.0504, -0.0422, -0.0405, -0.0535, -0.0608, -0.0927, -0.1254, -0.2026, -0.2971, -0.4185, -0.4674, -0.4463, -0.365, -0.3688, -0.3458, -0.3916, -0.7073, -1.0095, -1.3751, -1.6588, -1.8772, -2.0767, -2.3043, -2.6611, -3.1062, -3.5959, -4.219, -4.9769, -5.8233, -6.7525, -7.6786, -8.608, -9.5168, -10.5043, -11.5142, -12.5123, -13.5813, -14.7651, -16.0575, -17.3424, -18.6283, -19.6292, -20.4015, -20.9264, -21.2815, -21.5427, -21.7321, -22.1582, -22.6227, -23.2611, -23.9918, -24.8577, -25.9006, -26.9803, -28.0604, -29.1191, -30.1933, -31.3952, -32.7979, -34.2174, -35.3574, -36.699, -37.3798, -38.1833, -39.127, -39.9631, -40.7823, -41.4456, -41.9397, -42.228, -42.3921, -43.1224, -43.9887, -45.1821, -46.3106, -47.2983, -48.4335, -49.5782, -50.8779, -52.3791, -54.0956, -56.0031, -57.9376, -59.9907, -62.0781, -64.0315, -65.8572, -67.5321, -69.104, -70.5863, -71.8813, -73.3781, -74.8226, -76.2079, -77.5232, -78.7976, -80.0322, -81.3208, -82.6334, -83.9391, -85.2674, -86.6738, -88.051, -89.4385, -90.7849, -92.2035, -93.5643, -94.9323, -96.388, -97.8486, -99.2694, -100.7024, -102.165, -103.7395, -105.2959, -106.8726, -108.5303, -110.2054, -111.8196, -113.4236, -115.1003, -116.766, -118.4802, -120.1615, -121.8113, -123.6135, -125.3481, -127.1076, -128.8986, -130.7799, -132.453, -133.1538, -133.9272, -134.7752, -135.6537, -136.5009, -137.2615, -138.0572, -138.888, -139.6924, -140.3557, -141.0629, -141.853, -142.6362, -143.4364, -144.2991, -145.0251, -145.8074, -146.5912, -147.3189, -148.0547, -148.7896, -149.5185, -150.2393, -150.973, -151.7536, -152.5758, -153.3571, -154.1014, -154.7871, -155.436, -156.0332, -156.6181, -157.1861, -157.7823, -158.3975, -158.9895, -159.5506, -160.1073, -160.6873, -161.2504, -161.8023, -162.333, -162.8815, -163.4261, -163.9449, -164.4649, -165.0087, -165.5375, -166.0224, -166.4488, -166.8347, -167.2096, -167.5828, -167.9545, -168.3473, -168.7899, -169.2283, -169.6725, -170.1268, -170.6015, -171.0978, -171.6083, -172.1265, -172.6353, -173.1177, -173.5623, -173.9574, -174.3966, -174.8168, -175.3309, -175.83, -176.4875, -177.1209, -177.8425, -178.5892, -179.3835, -180.2425, -181.0357, -181.7195, -182.447, -183.1932, -183.8957, -184.6199, -185.3633, -186.1054, -186.7976, -187.4724, -188.0312, -188.6171, -189.1002, -189.5892, -190.1158, -190.6667, -191.2523, -191.92, -192.671, -193.4649, -194.3117, -195.2507, -196.317, -197.496, -198.8618, -200.3582, -201.9715, -203.7269, -205.618, -207.7093, -209.8073, -211.9753, -213.8289, -215.8472, -217.991, -220.1597, -222.3162, -223.8431, -225.1808, -226.0393, -226.6857, -227.253, -227.8503, -228.5243, -229.3545, -230.2377, -231.1321, -232.0608, -233.0077, -234.0677, -235.2144, -236.5558, -237.9632, -239.4215, -240.8719, -242.3782, -243.9402, -245.2728, -246.5012, -247.6797, -248.9961, -250.3158, -251.6307, -253.0166, -254.5635, -256.4041, -258.0749, -259.6066, -260.9438, -262.4176, -265.1393, -267.4421, -269.5406, -271.6047, -273.7723, -275.8481, -277.8838, -279.7824])

"""
plt.figure(figsize=(10, 5))  # 设置整体图形大小
plt.subplot(1, 2, 1)  # 创建1行2列的子图，并激活第1个子图
plt.plot(loss, marker='o')
plt.title('Data Plot 1')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 创建第二个子图
plt.subplot(1, 2, 2)  # 创建1行2列的子图，并激活第2个子图
plt.plot(control, marker='o')
plt.title('Data Plot 2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 保存图形为PNG格式文件
plt.savefig('two_plots.png')

# 显示图形
plt.show()

"""

min_index = np.argmin(loss)

print("最小值的下标是:", min_index)
print("最小值是:", loss[min_index])
print(control[min_index])
print(grad[min_index])
print(grad[50])

plt.figure(figsize=(15, 5))  # 设置整体图形大小
plt.subplot(1, 3, 1)  # 创建1行3列的子图，并激活第1个子图
plt.plot(loss, marker='o')
plt.title('Data Plot 1')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 创建第二个子图
plt.subplot(1, 3, 2)  # 创建1行3列的子图，并激活第2个子图
plt.plot(control, marker='o')
plt.title('Data Plot 2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 创建第三个子图
plt.subplot(1, 3, 3)  # 创建1行3列的子图，并激活第3个子图
plt.plot(grad, marker='o')
plt.title('Data Plot 3')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 保存图形为PNG格式文件
plt.savefig('three_plots.png')

# 显示图形
#plt.show()   


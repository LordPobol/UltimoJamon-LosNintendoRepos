# --------------------------------------------------
#
# Autor: Pablo Spínola López
# Description: Archivo de pruebas de la fase de entrenamiento con cobertura del 90%.
# 
# --------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import pytest
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from io import StringIO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from funciones.funcionesEntrenamiento import *

####################################################################################
# Simulación de DataFrame (Extraído de las 3 primeras entradas ds_tradicional.csv) #
DATA = """longitud_texto,num_palabras,comida,restriccion,purga,imagen_corporal,ejercicio,polaridad,subjetividad,tag_Anorexia,tag_Bulimia,tag_ED,tag_RexyBill,tag_Salud,tag_TCA,tag_Thinspo,tag_adelgazar,tag_alimentacionsaludable,tag_ana,tag_anamia,tag_anorexia,tag_anorexiaeetclub,tag_anorexic,tag_bulimia,tag_bulimianervosa,tag_bulimic,tag_bulimicgirl,tag_comida,tag_comidasaludable,tag_comidasana,tag_deporte,tag_desayuno,tag_dieta,tag_eatclean,tag_ed,tag_ejercicio,tag_entrenamiento,tag_fat,tag_fit,tag_food,tag_foodie,tag_foodporn,tag_gorda,tag_gym,tag_healthy,tag_healthyfood,tag_instafood,tag_lifestyle,tag_mia,tag_motivaciÃ³n,tag_motivation,tag_nutricion,tag_nutricionista,tag_perdergrasa,tag_perderpeso,tag_proana,tag_promia,tag_recetas,tag_salud,tag_saludable,tag_skinny,tag_tca,tag_thin,tag_thinspiration,tag_thinspo,tag_training,tag_vegan,tag_workout,tfidf_0,tfidf_1,tfidf_2,tfidf_3,tfidf_4,tfidf_5,tfidf_6,tfidf_7,tfidf_8,tfidf_9,tfidf_10,tfidf_11,tfidf_12,tfidf_13,tfidf_14,tfidf_15,tfidf_16,tfidf_17,tfidf_18,tfidf_19,tfidf_20,tfidf_21,tfidf_22,tfidf_23,tfidf_24,tfidf_25,tfidf_26,tfidf_27,tfidf_28,tfidf_29,tfidf_30,tfidf_31,tfidf_32,tfidf_33,tfidf_34,tfidf_35,tfidf_36,tfidf_37,tfidf_38,tfidf_39,tfidf_40,tfidf_41,tfidf_42,tfidf_43,tfidf_44,tfidf_45,tfidf_46,tfidf_47,tfidf_48,tfidf_49,tfidf_50,tfidf_51,tfidf_52,tfidf_53,tfidf_54,tfidf_55,tfidf_56,tfidf_57,tfidf_58,tfidf_59,tfidf_60,tfidf_61,tfidf_62,tfidf_63,tfidf_64,tfidf_65,tfidf_66,tfidf_67,tfidf_68,tfidf_69,tfidf_70,tfidf_71,tfidf_72,tfidf_73,tfidf_74,tfidf_75,tfidf_76,tfidf_77,tfidf_78,tfidf_79,tfidf_80,tfidf_81,tfidf_82,tfidf_83,tfidf_84,tfidf_85,tfidf_86,tfidf_87,tfidf_88,tfidf_89,tfidf_90,tfidf_91,tfidf_92,tfidf_93,tfidf_94,tfidf_95,tfidf_96,tfidf_97,tfidf_98,tfidf_99,tfidf_100,tfidf_101,tfidf_102,tfidf_103,tfidf_104,tfidf_105,tfidf_106,tfidf_107,tfidf_108,tfidf_109,tfidf_110,tfidf_111,tfidf_112,tfidf_113,tfidf_114,tfidf_115,tfidf_116,tfidf_117,tfidf_118,tfidf_119,tfidf_120,tfidf_121,tfidf_122,tfidf_123,tfidf_124,tfidf_125,tfidf_126,tfidf_127,tfidf_128,tfidf_129,tfidf_130,tfidf_131,tfidf_132,tfidf_133,tfidf_134,tfidf_135,tfidf_136,tfidf_137,tfidf_138,tfidf_139,tfidf_140,tfidf_141,tfidf_142,tfidf_143,tfidf_144,tfidf_145,tfidf_146,tfidf_147,tfidf_148,tfidf_149,tfidf_150,tfidf_151,tfidf_152,tfidf_153,tfidf_154,tfidf_155,tfidf_156,tfidf_157,tfidf_158,tfidf_159,tfidf_160,tfidf_161,tfidf_162,tfidf_163,tfidf_164,tfidf_165,tfidf_166,tfidf_167,tfidf_168,tfidf_169,tfidf_170,tfidf_171,tfidf_172,tfidf_173,tfidf_174,tfidf_175,tfidf_176,tfidf_177,tfidf_178,tfidf_179,tfidf_180,tfidf_181,tfidf_182,tfidf_183,tfidf_184,tfidf_185,tfidf_186,tfidf_187,tfidf_188,tfidf_189,tfidf_190,tfidf_191,tfidf_192,tfidf_193,tfidf_194,tfidf_195,tfidf_196,tfidf_197,tfidf_198,tfidf_199,tfidf_200,tfidf_201,tfidf_202,tfidf_203,tfidf_204,tfidf_205,tfidf_206,tfidf_207,tfidf_208,tfidf_209,tfidf_210,tfidf_211,tfidf_212,tfidf_213,tfidf_214,tfidf_215,tfidf_216,tfidf_217,tfidf_218,tfidf_219,tfidf_220,tfidf_221,tfidf_222,tfidf_223,tfidf_224,tfidf_225,tfidf_226,tfidf_227,tfidf_228,tfidf_229,tfidf_230,tfidf_231,tfidf_232,tfidf_233,tfidf_234,tfidf_235,tfidf_236,tfidf_237,tfidf_238,tfidf_239,tfidf_240,tfidf_241,tfidf_242,tfidf_243,tfidf_244,tfidf_245,tfidf_246,tfidf_247,tfidf_248,tfidf_249,tfidf_250,tfidf_251,tfidf_252,tfidf_253,tfidf_254,tfidf_255,tfidf_256,tfidf_257,tfidf_258,tfidf_259,tfidf_260,tfidf_261,tfidf_262,tfidf_263,tfidf_264,tfidf_265,tfidf_266,tfidf_267,tfidf_268,tfidf_269,tfidf_270,tfidf_271,tfidf_272,tfidf_273,tfidf_274,tfidf_275,tfidf_276,tfidf_277,tfidf_278,tfidf_279,tfidf_280,tfidf_281,tfidf_282,tfidf_283,tfidf_284,tfidf_285,tfidf_286,tfidf_287,tfidf_288,tfidf_289,tfidf_290,tfidf_291,tfidf_292,tfidf_293,tfidf_294,tfidf_295,tfidf_296,tfidf_297,tfidf_298,tfidf_299,tfidf_300,tfidf_301,tfidf_302,tfidf_303,tfidf_304,tfidf_305,tfidf_306,tfidf_307,tfidf_308,tfidf_309,tfidf_310,tfidf_311,tfidf_312,tfidf_313,tfidf_314,tfidf_315,tfidf_316,tfidf_317,tfidf_318,tfidf_319,tfidf_320,tfidf_321,tfidf_322,tfidf_323,tfidf_324,tfidf_325,tfidf_326,tfidf_327,tfidf_328,tfidf_329,tfidf_330,tfidf_331,tfidf_332,tfidf_333,tfidf_334,tfidf_335,tfidf_336,tfidf_337,tfidf_338,tfidf_339,tfidf_340,tfidf_341,tfidf_342,tfidf_343,tfidf_344,tfidf_345,tfidf_346,tfidf_347,tfidf_348,tfidf_349,tfidf_350,tfidf_351,tfidf_352,tfidf_353,tfidf_354,tfidf_355,tfidf_356,tfidf_357,tfidf_358,tfidf_359,tfidf_360,tfidf_361,tfidf_362,tfidf_363,tfidf_364,tfidf_365,tfidf_366,tfidf_367,tfidf_368,tfidf_369,tfidf_370,tfidf_371,tfidf_372,tfidf_373,tfidf_374,tfidf_375,tfidf_376,tfidf_377,tfidf_378,tfidf_379,tfidf_380,tfidf_381,tfidf_382,tfidf_383,tfidf_384,tfidf_385,tfidf_386,tfidf_387,tfidf_388,tfidf_389,tfidf_390,tfidf_391,tfidf_392,tfidf_393,tfidf_394,tfidf_395,tfidf_396,tfidf_397,tfidf_398,tfidf_399,tfidf_400,tfidf_401,tfidf_402,tfidf_403,tfidf_404,tfidf_405,tfidf_406,tfidf_407,tfidf_408,tfidf_409,tfidf_410,tfidf_411,tfidf_412,tfidf_413,tfidf_414,tfidf_415,tfidf_416,tfidf_417,tfidf_418,tfidf_419,tfidf_420,tfidf_421,tfidf_422,tfidf_423,tfidf_424,tfidf_425,tfidf_426,tfidf_427,tfidf_428,tfidf_429,tfidf_430,tfidf_431,tfidf_432,tfidf_433,tfidf_434,tfidf_435,tfidf_436,tfidf_437,tfidf_438,tfidf_439,tfidf_440,tfidf_441,tfidf_442,tfidf_443,tfidf_444,tfidf_445,tfidf_446,tfidf_447,tfidf_448,tfidf_449,tfidf_450,tfidf_451,tfidf_452,tfidf_453,tfidf_454,tfidf_455,tfidf_456,tfidf_457,tfidf_458,tfidf_459,tfidf_460,tfidf_461,tfidf_462,tfidf_463,tfidf_464,tfidf_465,tfidf_466,tfidf_467,tfidf_468,tfidf_469,tfidf_470,tfidf_471,tfidf_472,tfidf_473,tfidf_474,tfidf_475,tfidf_476,tfidf_477,tfidf_478,tfidf_479,tfidf_480,tfidf_481,tfidf_482,tfidf_483,tfidf_484,tfidf_485,tfidf_486,tfidf_487,tfidf_488,tfidf_489,tfidf_490,tfidf_491,tfidf_492,tfidf_493,tfidf_494,tfidf_495,tfidf_496,tfidf_497,tfidf_498,tfidf_499,tfidf_500,tfidf_501,tfidf_502,tfidf_503,tfidf_504,tfidf_505,tfidf_506,tfidf_507,tfidf_508,tfidf_509,tfidf_510,tfidf_511,tfidf_512,tfidf_513,tfidf_514,tfidf_515,tfidf_516,tfidf_517,tfidf_518,tfidf_519,tfidf_520,tfidf_521,tfidf_522,tfidf_523,tfidf_524,tfidf_525,tfidf_526,tfidf_527,tfidf_528,tfidf_529,tfidf_530,tfidf_531,tfidf_532,tfidf_533,tfidf_534,tfidf_535,tfidf_536,tfidf_537,tfidf_538,tfidf_539,tfidf_540,tfidf_541,tfidf_542,tfidf_543,tfidf_544,tfidf_545,tfidf_546,tfidf_547,tfidf_548,tfidf_549,tfidf_550,tfidf_551,tfidf_552,tfidf_553,tfidf_554,tfidf_555,tfidf_556,tfidf_557,tfidf_558,tfidf_559,tfidf_560,tfidf_561,tfidf_562,tfidf_563,tfidf_564,tfidf_565,tfidf_566,tfidf_567,tfidf_568,tfidf_569,tfidf_570,tfidf_571,tfidf_572,tfidf_573,tfidf_574,tfidf_575,tfidf_576,tfidf_577,tfidf_578,tfidf_579,tfidf_580,tfidf_581,tfidf_582,tfidf_583,tfidf_584,tfidf_585,tfidf_586,tfidf_587,tfidf_588,tfidf_589,tfidf_590,tfidf_591,tfidf_592,tfidf_593,tfidf_594,tfidf_595,tfidf_596,tfidf_597,tfidf_598,tfidf_599,tfidf_600,tfidf_601,tfidf_602,tfidf_603,tfidf_604,tfidf_605,tfidf_606,tfidf_607,tfidf_608,tfidf_609,tfidf_610,tfidf_611,tfidf_612,tfidf_613,tfidf_614,tfidf_615,tfidf_616,tfidf_617,tfidf_618,tfidf_619,tfidf_620,tfidf_621,tfidf_622,tfidf_623,tfidf_624,tfidf_625,tfidf_626,tfidf_627,tfidf_628,tfidf_629,tfidf_630,tfidf_631,tfidf_632,tfidf_633,tfidf_634,tfidf_635,tfidf_636,tfidf_637,tfidf_638,tfidf_639,tfidf_640,tfidf_641,tfidf_642,tfidf_643,tfidf_644,tfidf_645,tfidf_646,tfidf_647,tfidf_648,tfidf_649,tfidf_650,tfidf_651,tfidf_652,tfidf_653,tfidf_654,tfidf_655,tfidf_656,tfidf_657,tfidf_658,tfidf_659,tfidf_660,tfidf_661,tfidf_662,tfidf_663,tfidf_664,tfidf_665,tfidf_666,tfidf_667,tfidf_668,tfidf_669,tfidf_670,tfidf_671,tfidf_672,tfidf_673,tfidf_674,tfidf_675,tfidf_676,tfidf_677,tfidf_678,tfidf_679,tfidf_680,tfidf_681,tfidf_682,tfidf_683,tfidf_684,tfidf_685,tfidf_686,tfidf_687,tfidf_688,tfidf_689,tfidf_690,tfidf_691,tfidf_692,tfidf_693,tfidf_694,tfidf_695,tfidf_696,tfidf_697,tfidf_698,tfidf_699,tfidf_700,tfidf_701,tfidf_702,tfidf_703,tfidf_704,tfidf_705,tfidf_706,tfidf_707,tfidf_708,tfidf_709,tfidf_710,tfidf_711,tfidf_712,tfidf_713,tfidf_714,tfidf_715,tfidf_716,tfidf_717,tfidf_718,tfidf_719,tfidf_720,tfidf_721,tfidf_722,tfidf_723,tfidf_724,tfidf_725,tfidf_726,tfidf_727,tfidf_728,tfidf_729,tfidf_730,tfidf_731,tfidf_732,tfidf_733,tfidf_734,tfidf_735,tfidf_736,tfidf_737,tfidf_738,tfidf_739,tfidf_740,tfidf_741,tfidf_742,tfidf_743,tfidf_744,tfidf_745,tfidf_746,tfidf_747,tfidf_748,tfidf_749,tfidf_750,tfidf_751,tfidf_752,tfidf_753,tfidf_754,tfidf_755,tfidf_756,tfidf_757,tfidf_758,tfidf_759,tfidf_760,tfidf_761,tfidf_762,tfidf_763,tfidf_764,tfidf_765,tfidf_766,tfidf_767,tfidf_768,tfidf_769,tfidf_770,tfidf_771,tfidf_772,tfidf_773,tfidf_774,tfidf_775,tfidf_776,tfidf_777,tfidf_778,tfidf_779,tfidf_780,tfidf_781,tfidf_782,tfidf_783,tfidf_784,tfidf_785,tfidf_786,tfidf_787,tfidf_788,tfidf_789,tfidf_790,tfidf_791,tfidf_792,tfidf_793,tfidf_794,tfidf_795,tfidf_796,tfidf_797,tfidf_798,tfidf_799,tfidf_800,tfidf_801,tfidf_802,tfidf_803,tfidf_804,tfidf_805,tfidf_806,tfidf_807,tfidf_808,tfidf_809,tfidf_810,tfidf_811,tfidf_812,tfidf_813,tfidf_814,tfidf_815,tfidf_816,tfidf_817,tfidf_818,tfidf_819,tfidf_820,tfidf_821,tfidf_822,tfidf_823,tfidf_824,tfidf_825,tfidf_826,tfidf_827,tfidf_828,tfidf_829,tfidf_830,tfidf_831,tfidf_832,tfidf_833,tfidf_834,tfidf_835,tfidf_836,tfidf_837,tfidf_838,tfidf_839,tfidf_840,tfidf_841,tfidf_842,tfidf_843,tfidf_844,tfidf_845,tfidf_846,tfidf_847,tfidf_848,tfidf_849,tfidf_850,tfidf_851,tfidf_852,tfidf_853,tfidf_854,tfidf_855,tfidf_856,tfidf_857,tfidf_858,tfidf_859,tfidf_860,tfidf_861,tfidf_862,tfidf_863,tfidf_864,tfidf_865,tfidf_866,tfidf_867,tfidf_868,tfidf_869,tfidf_870,tfidf_871,tfidf_872,tfidf_873,tfidf_874,tfidf_875,tfidf_876,tfidf_877,tfidf_878,tfidf_879,tfidf_880,tfidf_881,tfidf_882,tfidf_883,tfidf_884,tfidf_885,tfidf_886,tfidf_887,tfidf_888,tfidf_889,tfidf_890,tfidf_891,tfidf_892,tfidf_893,tfidf_894,tfidf_895,tfidf_896,tfidf_897,tfidf_898,tfidf_899,tfidf_900,tfidf_901,tfidf_902,tfidf_903,tfidf_904,tfidf_905,tfidf_906,tfidf_907,tfidf_908,tfidf_909,tfidf_910,tfidf_911,tfidf_912,tfidf_913,tfidf_914,tfidf_915,tfidf_916,tfidf_917,tfidf_918,tfidf_919,tfidf_920,tfidf_921,tfidf_922,tfidf_923,tfidf_924,tfidf_925,tfidf_926,tfidf_927,tfidf_928,tfidf_929,tfidf_930,tfidf_931,tfidf_932,tfidf_933,tfidf_934,tfidf_935,tfidf_936,tfidf_937,tfidf_938,tfidf_939,tfidf_940,tfidf_941,tfidf_942,tfidf_943,tfidf_944,tfidf_945,tfidf_946,tfidf_947,tfidf_948,tfidf_949,tfidf_950,tfidf_951,tfidf_952,tfidf_953,tfidf_954,tfidf_955,tfidf_956,tfidf_957,tfidf_958,tfidf_959,tfidf_960,tfidf_961,tfidf_962,tfidf_963,tfidf_964,tfidf_965,tfidf_966,tfidf_967,tfidf_968,tfidf_969,tfidf_970,tfidf_971,tfidf_972,tfidf_973,tfidf_974,tfidf_975,tfidf_976,tfidf_977,tfidf_978,tfidf_979,tfidf_980,tfidf_981,tfidf_982,tfidf_983,tfidf_984,tfidf_985,tfidf_986,tfidf_987,tfidf_988,tfidf_989,tfidf_990,tfidf_991,tfidf_992,tfidf_993,tfidf_994,tfidf_995,tfidf_996,tfidf_997,tfidf_998,tfidf_999,tfidf_1000,tfidf_1001,tfidf_1002,tfidf_1003,tfidf_1004,tfidf_1005,tfidf_1006,tfidf_1007,tfidf_1008,tfidf_1009,tfidf_1010,tfidf_1011,tfidf_1012,tfidf_1013,tfidf_1014,tfidf_1015,tfidf_1016,tfidf_1017,tfidf_1018,tfidf_1019,tfidf_1020,tfidf_1021,tfidf_1022,tfidf_1023,tfidf_1024,tfidf_1025,tfidf_1026,tfidf_1027,tfidf_1028,tfidf_1029,tfidf_1030,tfidf_1031,tfidf_1032,tfidf_1033,tfidf_1034,tfidf_1035,tfidf_1036,tfidf_1037,tfidf_1038,tfidf_1039,tfidf_1040,tfidf_1041,tfidf_1042,tfidf_1043,tfidf_1044,tfidf_1045,tfidf_1046,tfidf_1047,tfidf_1048,tfidf_1049,tfidf_1050,tfidf_1051,tfidf_1052,tfidf_1053,tfidf_1054,tfidf_1055,tfidf_1056,tfidf_1057,tfidf_1058,tfidf_1059,tfidf_1060,tfidf_1061,tfidf_1062,tfidf_1063,tfidf_1064,tfidf_1065,tfidf_1066,tfidf_1067,tfidf_1068,tfidf_1069,tfidf_1070,tfidf_1071,tfidf_1072,tfidf_1073,tfidf_1074,tfidf_1075,tfidf_1076,tfidf_1077,tfidf_1078,tfidf_1079,tfidf_1080,tfidf_1081,tfidf_1082,tfidf_1083,tfidf_1084,tfidf_1085,tfidf_1086,tfidf_1087,tfidf_1088,tfidf_1089,tfidf_1090,tfidf_1091,tfidf_1092,tfidf_1093,tfidf_1094,tfidf_1095,tfidf_1096,tfidf_1097,tfidf_1098,tfidf_1099,tfidf_1100,tfidf_1101,tfidf_1102,tfidf_1103,tfidf_1104,tfidf_1105,tfidf_1106,tfidf_1107,tfidf_1108,tfidf_1109,tfidf_1110,tfidf_1111,tfidf_1112,tfidf_1113,tfidf_1114,tfidf_1115,tfidf_1116,tfidf_1117,tfidf_1118,tfidf_1119,tfidf_1120,tfidf_1121,tfidf_1122,tfidf_1123,tfidf_1124,tfidf_1125,tfidf_1126,tfidf_1127,tfidf_1128,tfidf_1129,tfidf_1130,tfidf_1131,tfidf_1132,tfidf_1133,tfidf_1134,tfidf_1135,tfidf_1136,tfidf_1137,tfidf_1138,tfidf_1139,tfidf_1140,tfidf_1141,tfidf_1142,tfidf_1143,tfidf_1144,tfidf_1145,tfidf_1146,tfidf_1147,tfidf_1148,tfidf_1149,tfidf_1150,tfidf_1151,tfidf_1152,tfidf_1153,tfidf_1154,tfidf_1155,tfidf_1156,tfidf_1157,tfidf_1158,tfidf_1159,tfidf_1160,tfidf_1161,tfidf_1162,tfidf_1163,tfidf_1164,tfidf_1165,tfidf_1166,tfidf_1167,tfidf_1168,tfidf_1169,tfidf_1170,tfidf_1171,tfidf_1172,tfidf_1173,tfidf_1174,tfidf_1175,tfidf_1176,tfidf_1177,tfidf_1178,tfidf_1179,tfidf_1180,tfidf_1181,tfidf_1182,tfidf_1183,tfidf_1184,tfidf_1185,tfidf_1186,tfidf_1187,tfidf_1188,tfidf_1189,tfidf_1190,tfidf_1191,tfidf_1192,tfidf_1193,tfidf_1194,tfidf_1195,tfidf_1196,tfidf_1197,tfidf_1198,tfidf_1199,tfidf_1200,tfidf_1201,tfidf_1202,tfidf_1203,tfidf_1204,tfidf_1205,tfidf_1206,tfidf_1207,tfidf_1208,tfidf_1209,tfidf_1210,tfidf_1211,tfidf_1212,tfidf_1213,tfidf_1214,tfidf_1215,tfidf_1216,tfidf_1217,tfidf_1218,tfidf_1219,tfidf_1220,tfidf_1221,tfidf_1222,tfidf_1223,tfidf_1224,tfidf_1225,tfidf_1226,tfidf_1227,tfidf_1228,tfidf_1229,tfidf_1230,tfidf_1231,tfidf_1232,tfidf_1233,tfidf_1234,tfidf_1235,tfidf_1236,tfidf_1237,tfidf_1238,tfidf_1239,tfidf_1240,tfidf_1241,tfidf_1242,tfidf_1243,tfidf_1244,tfidf_1245,tfidf_1246,tfidf_1247,tfidf_1248,tfidf_1249,tfidf_1250,tfidf_1251,tfidf_1252,tfidf_1253,tfidf_1254,tfidf_1255,tfidf_1256,tfidf_1257,tfidf_1258,tfidf_1259,tfidf_1260,tfidf_1261,tfidf_1262,tfidf_1263,tfidf_1264,tfidf_1265,tfidf_1266,tfidf_1267,tfidf_1268,tfidf_1269,tfidf_1270,tfidf_1271,tfidf_1272,tfidf_1273,tfidf_1274,tfidf_1275,tfidf_1276,tfidf_1277,tfidf_1278,tfidf_1279,tfidf_1280,tfidf_1281,tfidf_1282,tfidf_1283,tfidf_1284,tfidf_1285,tfidf_1286,tfidf_1287,tfidf_1288,tfidf_1289,tfidf_1290,tfidf_1291,tfidf_1292,tfidf_1293,tfidf_1294,tfidf_1295,tfidf_1296,tfidf_1297,tfidf_1298,tfidf_1299,class
0.24219111576013277,-0.059670749193371515,0,0,0,0,0,1.173737999324584,0.3534199799626798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5917119611094862,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6858422284053577,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4236713264028367,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0
-1.5499849540623942,-1.624909887563395,0,0,0,0,0,-0.33636701569099176,-1.1281508021871602,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1
-0.4564537928147507,-0.28327634038908917,3,2,0,0,0,0.79621174557069,0.9460482928226157,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.46863656256275027,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2601274709028979,0.43277901202809915,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35804133779979747,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3713252340054927,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3915191592334458,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.32565705401456746,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0"""

@pytest.fixture
def mock_df():
    return pd.read_csv(StringIO(DATA))
####################################################################################
####################################################################################

""" Pruebas unitarias para las funciones de entrenamiento de modelos """

##########################################################################
# Pruebas unitarias para la primera función: cargar_datos_entrenamiento()#
##########################################################################

def test_cargar_datos_entrenamiento(monkeypatch, mock_df):
    def mock_read_csv(path):
        return mock_df
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    X, y = cargar_datos_entrenamiento()
    assert "class" not in X.columns
    assert y.tolist() == [0, 1, 0]

def test_cargar_datos_entrenamiento_forma(monkeypatch, mock_df):
    def mock_read_csv(path):
        return mock_df
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    X, y = cargar_datos_entrenamiento()
    assert 1368 == X.shape[1]
    assert 3 == X.shape[0]

###################################################################
# Pruebas unitarias para la segunda función: cargar_datos_prueba()#
###################################################################

def test_cargar_datos_prueba(monkeypatch, mock_df):
    def mock_read_csv(path):
        return mock_df
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    X, y = cargar_datos_prueba()
    assert "class" not in X.columns
    assert y.tolist() == [0, 1, 0]

def test_cargar_datos_prueba_forma(monkeypatch, mock_df):
    def mock_read_csv(path):
        return mock_df
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    X, y = cargar_datos_prueba()
    assert 1368 == X.shape[1]
    assert 3 == X.shape[0]

#######################################################################
# Pruebas unitarias para la tercera función: imprimir_forma(DataFrame)#
#######################################################################

def test_imprimir_forma_pequeña():
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6],
        "class": [0, 1, 0, 1, 1, 0]
    })
    shape, head = imprimir_forma(df)
    assert shape == (6, 2)
    assert head.shape == (5, 2)

def test_imprimir_forma_grande(monkeypatch, mock_df):
    def mock_read_csv(path):
        return mock_df
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    X, _ = cargar_datos_prueba()
    shape, head = imprimir_forma(X)
    assert shape == (3, 1368)
    assert head.shape == (3, 1368)

##################################################################################
# Pruebas unitarias para la cuarta función: division_train_val(DataFrame, Series)#
##################################################################################

def test_division_train_val_shapes():
    X = pd.DataFrame({'a': range(100)})
    y = pd.Series([0]*50 + [1]*50)
    X_train, X_val, y_train, y_val = division_train_val(X, y)

    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(y_train) == 80
    assert len(y_val) == 20

def test_division_train_val_stratification():
    X = pd.DataFrame({'a': range(100)})
    y = pd.Series([0]*70 + [1]*30)
    _, _, y_train, y_val = division_train_val(X, y)

    assert y_train.value_counts(normalize=True).round(1).tolist() == [0.7, 0.3]
    assert y_val.value_counts(normalize=True).round(1).tolist() == [0.7, 0.3]

#############################################################################################
# Pruebas unitarias para la quinta función: reporte_clasificacion(DataFrame, Series, modelo)#
#############################################################################################

def test_reporte_clasificacion_proba():
    X, y = make_classification(n_samples=100, n_features=5, random_state=1)
    model = RandomForestClassifier(random_state=1)
    model.fit(X, y)
    
    y_pred, y_res, reporte = reporte_clasificacion(X, y, model)
    
    assert len(y_pred) == len(y)
    assert len(y_res) == len(y)
    assert isinstance(reporte, str)
    assert "precision" in reporte.lower()

def test_reporte_clasificacion_lineal():
    X, y = make_classification(n_samples=100, n_features=5, random_state=1)
    model = LogisticRegression()
    model.fit(X, y)
    
    y_pred, y_res, reporte = reporte_clasificacion(X, y, model, lineal=True)
    
    assert len(y_pred) == len(y)
    assert len(y_res) == len(y)
    assert isinstance(reporte, str)
    assert "recall" in reporte.lower()

##################################################################################
# Pruebas unitarias para la sexta función: crear_matriz_confusion(Series, Series)#
##################################################################################

def test_crear_matriz_confusion_basica():
    y_test = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    cm, disp = crear_matriz_confusion(y_test, y_pred)
    assert cm.shape == (2, 2)
    assert isinstance(disp, ConfusionMatrixDisplay)
    assert cm[0][0] == 2  # TN (True negatives)
    assert cm[1][1] == 2  # TP (True positives)

def test_crear_matriz_confusion_binaria_perfecta():
    y_test = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    cm, _ = crear_matriz_confusion(y_test, y_pred)
    assert np.array_equal(cm, np.array([[2, 0], [0, 2]])) # Aquí se forma una matriz perfecta, sin falsos

##############################################################################
# Pruebas unitarias para la séptima función: calcular_roc_auc(Series, Series)#
##############################################################################

def test_calcular_roc_auc_resultados_correctos():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds, auc_score = calcular_roc_auc(y_true, y_scores)
    assert isinstance(fpr, np.ndarray)
    assert isinstance(tpr, np.ndarray)
    assert isinstance(thresholds, np.ndarray)
    assert isinstance(auc_score, float)
    assert 0 <= auc_score <= 1

def test_calcular_roc_auc_auc_valido():
    y_true = [0, 1, 1, 0, 1]
    y_scores = [0.1, 0.9, 0.8, 0.4, 0.6]
    _, _, _, auc_score = calcular_roc_auc(y_true, y_scores)
    assert round(auc_score, 2) == round(roc_auc_score(y_true, y_scores), 2)

##############################################################################
# Pruebas unitarias para la octava función: metricas_tpr_fpr(ConfusionMatrix)#
##############################################################################

def test_metricas_tpr_fpr_valores_correctos():
    cm = np.array([[50, 10],
                   [5, 35]])      # TN=50, FP=10, FN=5, TP=35
    TPR, FPR = metricas_tpr_fpr(cm)
    assert round(TPR, 2) == 0.88  # 35 / (35 + 5)
    assert round(FPR, 2) == 0.17  # 10 / (10 + 50)

def test_metricas_tpr_fpr_edge_case():
    cm = np.array([[0, 0],
                   [0, 1]])  # TN=0, FP=0, FN=0, TP=1
    TPR, FPR = metricas_tpr_fpr(cm)
    assert TPR == 1.0
    assert FPR == 0.0

###########################################################################
# Pruebas unitarias para la novena función: hacer_pepinillo(model, string)#
###########################################################################

def test_hacer_pepinillo_crea_archivo_temporal(tmp_path):
    modelo = LogisticRegression()
    ruta_modelo = tmp_path / "modelo.pkl"
    
    hacer_pepinillo(modelo, str(ruta_modelo), test=True)
    
    assert ruta_modelo.exists()

def test_hacer_pepinillo_modelo_valido(tmp_path):
    modelo = LogisticRegression()
    ruta_modelo = tmp_path / "modelo_valido.pkl"
    
    hacer_pepinillo(modelo, str(ruta_modelo), test=True)
    
    with open(ruta_modelo, "rb") as f:
        modelo_cargado = pickle.load(f)
    
    assert isinstance(modelo_cargado, LogisticRegression)
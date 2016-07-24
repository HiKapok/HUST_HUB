TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
#CONFIG -= qt

SOURCES += main.cpp \
    kcalchist.cpp \
    kutility.cpp

#QT       += core

#QT       -= gui

#INCLUDEPATH += D:/QT/Libs/QtOpenCV3_ST/include
#LIBS += -LD:/QT/Libs/QtOpenCV3_ST/lib\
#-lopencv_calib3d310\
#-lopencv_core310\
#-lopencv_features2d310\
#-lopencv_flann310\
#-lopencv_highgui310\
#-lopencv_imgcodecs310\
#-lopencv_imgproc310\
#-lopencv_ml310\
#-lopencv_objdetect310\
#-lopencv_photo310\
#-lopencv_shape310\
#-lopencv_stitching310\
#-lopencv_superres310\
#-lopencv_ts310\
#-lopencv_video310\
#-lopencv_videoio310
#INCLUDEPATH += D:/QT/Libs/QtOpenCV31/include
#LIBS += -LD:/QT/Libs/QtOpenCV31/lib\
#-llibopencv_calib3d310\
#-llibopencv_core310\
#-llibopencv_features2d310\
#-llibopencv_flann310\
#-llibopencv_highgui310\
#-llibopencv_imgcodecs310\
#-llibopencv_imgproc310\
#-llibopencv_ml310\
#-llibopencv_objdetect310\
#-llibopencv_photo310\
#-llibopencv_shape310\
#-llibopencv_stitching310\
#-llibopencv_superres310\
#-llibopencv_videostab310\
#-llibopencv_video310\
#-llibopencv_videoio310\
#-lopencv_ts310
#edited for openCV3.0
#库引入方�#INCLUDEPATH += E:/QTPrj/QtOpenCV3/include
#LIBS += -LE:/QTPrj/QtOpenCV3/lib\
INCLUDEPATH += D:/QT/Libs/QtOpenCV3/include
LIBS += -LD:/QT/Libs/QtOpenCV3/lib\
-llibopencv_calib3d300\
-llibopencv_core300\
-llibopencv_features2d300\
-llibopencv_flann300\
-llibopencv_highgui300\
-llibopencv_imgcodecs300\
-llibopencv_imgproc300\
-llibopencv_ml300\
-llibopencv_objdetect300\
-llibopencv_photo300\
-llibopencv_shape300\
-llibopencv_stitching300\
-llibopencv_superres300\
-lopencv_hal300
#库引入方�#INCLUDEPATH += E:/QTPrj/QtOpenCV3/include
#LIBS += -L E:/QTPrj/QtOpenCV3/lib/libopencv_*.a

#edited for boost1.6.0
#static library
#DEFINES += BOOST_THREAD_USE_LIB

##INCLUDEPATH += E:/boost_1_60_0/QtLib/include
##LIBS += -LE:/boost_1_60_0/QtLib/lib\
#INCLUDEPATH += D:/Qt/QtLibs/Boost/include
#LIBS += -LD:/Qt/QtLibs/Boost/lib\
#-lboost_atomic-mgw49-mt-s-1_60\
#-lboost_atomic-mgw49-mt-sd-1_60\
#-lboost_chrono-mgw49-mt-s-1_60\
#-lboost_chrono-mgw49-mt-sd-1_60\
#-lboost_container-mgw49-mt-s-1_60\
#-lboost_container-mgw49-mt-sd-1_60\
#-lboost_context-mgw49-mt-s-1_60\
#-lboost_context-mgw49-mt-sd-1_60\
#-lboost_coroutine-mgw49-mt-s-1_60\
#-lboost_coroutine-mgw49-mt-sd-1_60\
#-lboost_date_time-mgw49-mt-s-1_60\
#-lboost_date_time-mgw49-mt-sd-1_60\
#-lboost_exception-mgw49-mt-s-1_60\
#-lboost_exception-mgw49-mt-sd-1_60\
#-lboost_filesystem-mgw49-mt-s-1_60\
#-lboost_filesystem-mgw49-mt-sd-1_60\
#-lboost_iostreams-mgw49-mt-s-1_60\
#-lboost_iostreams-mgw49-mt-sd-1_60\
#-lboost_locale-mgw49-mt-s-1_60\
#-lboost_locale-mgw49-mt-sd-1_60\
#-lboost_log-mgw49-mt-s-1_60\
#-lboost_log-mgw49-mt-sd-1_60\
#-lboost_log_setup-mgw49-mt-s-1_60\
#-lboost_log_setup-mgw49-mt-sd-1_60\
#-lboost_math_c99-mgw49-mt-s-1_60\
#-lboost_math_c99-mgw49-mt-sd-1_60\
#-lboost_math_c99f-mgw49-mt-s-1_60\
#-lboost_math_c99f-mgw49-mt-sd-1_60\
#-lboost_math_c99l-mgw49-mt-s-1_60\
#-lboost_math_c99l-mgw49-mt-sd-1_60\
#-lboost_math_tr1-mgw49-mt-s-1_60\
#-lboost_math_tr1-mgw49-mt-sd-1_60\
#-lboost_math_tr1f-mgw49-mt-s-1_60\
#-lboost_math_tr1f-mgw49-mt-sd-1_60\
#-lboost_math_tr1l-mgw49-mt-s-1_60\
#-lboost_math_tr1l-mgw49-mt-sd-1_60\
#-lboost_prg_exec_monitor-mgw49-mt-s-1_60\
#-lboost_prg_exec_monitor-mgw49-mt-sd-1_60\
#-lboost_program_options-mgw49-mt-s-1_60\
#-lboost_program_options-mgw49-mt-sd-1_60\
#-lboost_python-mgw49-mt-s-1_60\
#-lboost_python-mgw49-mt-sd-1_60\
#-lboost_python3-mgw49-mt-s-1_60\
#-lboost_python3-mgw49-mt-sd-1_60\
#-lboost_random-mgw49-mt-s-1_60\
#-lboost_random-mgw49-mt-sd-1_60\
#-lboost_regex-mgw49-mt-s-1_60\
#-lboost_regex-mgw49-mt-sd-1_60\
#-lboost_serialization-mgw49-mt-s-1_60\
#-lboost_serialization-mgw49-mt-sd-1_60\
#-lboost_signals-mgw49-mt-s-1_60\
#-lboost_signals-mgw49-mt-sd-1_60\
#-lboost_system-mgw49-mt-s-1_60\
#-lboost_system-mgw49-mt-sd-1_60\
#-lboost_test_exec_monitor-mgw49-mt-s-1_60\
#-lboost_test_exec_monitor-mgw49-mt-sd-1_60\
#-lboost_thread-mgw49-mt-s-1_60\
#-lboost_thread-mgw49-mt-sd-1_60\
#-lboost_timer-mgw49-mt-s-1_60\
#-lboost_timer-mgw49-mt-sd-1_60\
#-lboost_type_erasure-mgw49-mt-s-1_60\
#-lboost_type_erasure-mgw49-mt-sd-1_60\
#-lboost_unit_test_framework-mgw49-mt-s-1_60\
#-lboost_unit_test_framework-mgw49-mt-sd-1_60\
#-lboost_wserialization-mgw49-mt-s-1_60\
#-lboost_wserialization-mgw49-mt-sd-1_60
#dynamic library
#-lboost_atomic-mgw49-mt-1_60\
#-lboost_atomic-mgw49-mt-d-1_60\
#-lboost_chrono-mgw49-mt-1_60\
#-lboost_chrono-mgw49-mt-d-1_60\
#-lboost_container-mgw49-mt-1_60\
#-lboost_container-mgw49-mt-d-1_60\
#-lboost_context-mgw49-mt-1_60\
#-lboost_context-mgw49-mt-d-1_60\
#-lboost_coroutine-mgw49-mt-1_60\
#-lboost_coroutine-mgw49-mt-d-1_60\
#-lboost_date_time-mgw49-mt-1_60\
#-lboost_date_time-mgw49-mt-d-1_60\
#-lboost_exception-mgw49-mt-1_60\
#-lboost_exception-mgw49-mt-d-1_60\
#-lboost_filesystem-mgw49-mt-1_60\
#-lboost_filesystem-mgw49-mt-d-1_60\
#-lboost_iostreams-mgw49-mt-1_60\
#-lboost_iostreams-mgw49-mt-d-1_60\
#-lboost_locale-mgw49-mt-1_60\
#-lboost_locale-mgw49-mt-d-1_60\
#-lboost_log-mgw49-mt-1_60\
#-lboost_log-mgw49-mt-d-1_60\
#-lboost_log_setup-mgw49-mt-1_60\
#-lboost_log_setup-mgw49-mt-d-1_60\
#-lboost_math_c99-mgw49-mt-1_60\
#-lboost_math_c99-mgw49-mt-d-1_60\
#-lboost_math_c99f-mgw49-mt-1_60\
#-lboost_math_c99f-mgw49-mt-d-1_60\
#-lboost_math_c99l-mgw49-mt-1_60\
#-lboost_math_c99l-mgw49-mt-d-1_60\
#-lboost_math_tr1-mgw49-mt-1_60\
#-lboost_math_tr1-mgw49-mt-d-1_60\
#-lboost_math_tr1f-mgw49-mt-1_60\
#-lboost_math_tr1f-mgw49-mt-d-1_60\
#-lboost_math_tr1l-mgw49-mt-1_60\
#-lboost_math_tr1l-mgw49-mt-d-1_60\
#-lboost_prg_exec_monitor-mgw49-mt-1_60\
#-lboost_prg_exec_monitor-mgw49-mt-d-1_60\
#-lboost_program_options-mgw49-mt-1_60\
#-lboost_program_options-mgw49-mt-d-1_60\
#-lboost_python-mgw49-mt-1_60\
#-lboost_python-mgw49-mt-d-1_60\
#-lboost_python3-mgw49-mt-1_60\
#-lboost_python3-mgw49-mt-d-1_60\
#-lboost_random-mgw49-mt-1_60\
#-lboost_random-mgw49-mt-d-1_60\
#-lboost_regex-mgw49-mt-1_60\
#-lboost_regex-mgw49-mt-d-1_60\
#-lboost_serialization-mgw49-mt-1_60\
#-lboost_serialization-mgw49-mt-d-1_60\
#-lboost_signals-mgw49-mt-1_60\
#-lboost_signals-mgw49-mt-d-1_60\
#-lboost_system-mgw49-mt-1_60\
#-lboost_system-mgw49-mt-d-1_60\
#-lboost_test_exec_monitor-mgw49-mt-1_60\
#-lboost_test_exec_monitor-mgw49-mt-d-1_60\
#-lboost_thread-mgw49-mt-1_60\
#-lboost_thread-mgw49-mt-d-1_60\
#-lboost_timer-mgw49-mt-1_60\
#-lboost_timer-mgw49-mt-d-1_60\
#-lboost_type_erasure-mgw49-mt-1_60\
#-lboost_type_erasure-mgw49-mt-d-1_60\
#-lboost_unit_test_framework-mgw49-mt-1_60\
#-lboost_unit_test_framework-mgw49-mt-d-1_60\
#-lboost_wserialization-mgw49-mt-1_60\
#-lboost_wserialization-mgw49-mt-d-1_60

##edited for GDAL200
#INCLUDEPATH += E:/QTPrj/GraduationDesignTest/GDAL200/include
#LIBS += -LE:/QTPrj/GraduationDesignTest/GDAL200/lib\
#-lgdal
##LIBS += E:/QTPrj/GraduationDesignTest/GDAL200/lib/libgdal.a

##edited for tinyxml
#INCLUDEPATH += E:/QTPrj/GraduationDesignTest/QtTinyxml/include
#LIBS += -LE:/QTPrj/GraduationDesignTest/QtTinyxml/lib\
#-ltinyxml

##edited for libsvm
##库包含方�#INCLUDEPATH += E:/QTPrj/GraduationDesignTest/QtLibsvm/include
#LIBS += -LE:/QTPrj/GraduationDesignTest/QtLibsvm/lib\
#-lsvm

#edited for GDAL200
#INCLUDEPATH += D:/Qt/QtPrj/GraduationDesignTest/GDAL200/include
#LIBS += -LD:/Qt/QtPrj/GraduationDesignTest/GDAL200/lib\
#-lgdal
#LIBS += E:/QTPrj/GraduationDesignTest/GDAL200/lib/libgdal.a

#edited for tinyxml
#INCLUDEPATH += D:/Qt/QtPrj/GraduationDesignTest/QtTinyxml/include
#LIBS += -LD:/Qt/QtPrj/GraduationDesignTest/QtTinyxml/lib\
#-ltinyxml

#edited for libsvm
#库包含方�#INCLUDEPATH += D:/Qt/QtPrj/GraduationDesignTest/QtLibsvm/include
#LIBS += -LD:/Qt/QtPrj/GraduationDesignTest/QtLibsvm/lib\
#-lsvm

#库包含方�#INCLUDEPATH += E:/QTPrj/QtLibsvm/include
#LIBS += E:/QTPrj/QtLibsvm/lib/svm.o

HEADERS += \
    kcalchist.h \
    kutility.h

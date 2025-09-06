#include <QCoreApplication>
#include <QDebug>
#include <H5Cpp.h>

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    qDebug() << "[DBG] Qt + HDF5 demo starting. Args:" << app.arguments();

    unsigned maj=0, min=0, rel=0;
    H5get_libversion(&maj, &min, &rel);
    qDebug() << "[DBG] HDF5 version =" << maj << "." << min << "." << rel;

    try {
        const QString file = "qt_test.h5";
        qDebug() << "[DBG] Creating file" << file;
        H5::H5File f(file.toStdString(), H5F_ACC_TRUNC);

        int value = 7;
        hsize_t dims[1] = {1};
        H5::DataSpace space(1, dims);

        qDebug() << "[DBG] Creating dataset 'seven'...";
        H5::DataSet ds = f.createDataSet("seven", H5::PredType::NATIVE_INT, space);

        qDebug() << "[DBG] Writing value=7 ...";
        ds.write(&value, H5::PredType::NATIVE_INT);

        int back = 0;
        qDebug() << "[DBG] Reading back...";
        ds.read(&back, H5::PredType::NATIVE_INT);
        qDebug() << "[DBG] Read-back =" << back;

        const hsize_t nobj = f.getNumObjs();
        qDebug() << "[DBG] Root object count =" << nobj;
        for (hsize_t i = 0; i < nobj; ++i) {
            qDebug() << "[DBG] -" << f.getObjnameByIdx(i).c_str();
        }
    }
    catch (const H5::Exception& e) {
        qWarning() << "[ERR] HDF5:" << e.getCDetailMsg();
        return 1;
    }
    catch (const std::exception& e) {
        qWarning() << "[ERR] std:" << e.what();
        return 2;
    }

    qDebug() << "[DBG] Done.";
    return 0;
}

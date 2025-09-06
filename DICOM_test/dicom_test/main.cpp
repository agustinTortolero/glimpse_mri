#include <QCoreApplication>
#include <QDebug>
#include <vector>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcuid.h>

static void putOF(DcmDataset* ds, const DcmTagKey& key, const char* val) {
    OFCondition st = ds->putAndInsertString(key, val);
    if (st.bad()) throw std::runtime_error(QString("[ERR] put %1: %2")
                                     .arg(key.toString().c_str()).arg(st.text()).toStdString());
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qDebug() << "[DBG] dcmtk_test starting";

    try {
        const int rows=128, cols=128;
        std::vector<quint16> px(rows*cols);
        for (int r=0;r<rows;++r) for (int c=0;c<cols;++c) px[r*cols+c] = (quint16)((r*c)%4096);

        DcmFileFormat ff; auto* ds = ff.getDataset();
        char studyUID[100];  dcmGenerateUniqueIdentifier(studyUID, SITE_STUDY_UID_ROOT);
        char seriesUID[100]; dcmGenerateUniqueIdentifier(seriesUID, SITE_SERIES_UID_ROOT);
        char instUID[100];   dcmGenerateUniqueIdentifier(instUID,  SITE_INSTANCE_UID_ROOT);

        putOF(ds, DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
        putOF(ds, DCM_SOPInstanceUID, instUID);
        putOF(ds, DCM_StudyInstanceUID, studyUID);
        putOF(ds, DCM_SeriesInstanceUID, seriesUID);
        putOF(ds, DCM_PatientName, "Test^Patient");
        putOF(ds, DCM_PatientID, "TEST001");
        putOF(ds, DCM_Modality, "OT");
        putOF(ds, DCM_PhotometricInterpretation, "MONOCHROME2");
        ds->putAndInsertUint16(DCM_Rows, rows);
        ds->putAndInsertUint16(DCM_Columns, cols);
        ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
        ds->putAndInsertUint16(DCM_BitsAllocated, 16);
        ds->putAndInsertUint16(DCM_BitsStored, 12);
        ds->putAndInsertUint16(DCM_HighBit, 11);
        ds->putAndInsertUint16(DCM_PixelRepresentation, 0);

        OFCondition st = ds->putAndInsertUint8Array(DCM_PixelData,
                                                    reinterpret_cast<const Uint8*>(px.data()),
                                                    static_cast<Uint32>(px.size()*sizeof(quint16)));
        if (st.bad()) throw std::runtime_error(st.text());

        qDebug() << "[DBG] Saving dcmtk_test.dcm";
        st = ff.saveFile("dcmtk_test.dcm", EXS_LittleEndianExplicit);
        if (st.bad()) throw std::runtime_error(st.text());

        qDebug() << "[DBG] OK";
        return 0;
    } catch (const std::exception& e) {
        qWarning() << e.what();
        return 1;
    }
}

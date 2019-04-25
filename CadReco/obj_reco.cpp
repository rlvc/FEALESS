#include "obj_reco.h"
#include "obj_reco_lmicp.h"
////#include "log_def.h"
//#include <iosfwd>
//#define LIB_VERSION "3.1.1"
//string CObjReco::GetVersion()
//{
//    std::stringstream s;
//    s << "Lenovo 3D Object Recognition. Version " << LIB_VERSION " Compile Time: " __DATE__ " " << __TIME__;
//    return s.str();
//}

CObjReco * CObjReco::Create(EObjRecoType eType)
{
    switch (eType)
    {
        case EObjReco_FEATURE:
//            return new CObjRecoFeature();
            break;
        case EObjReco_LmICP:
            return new CObjRecoLmICP();
        case EObjReco_BB8:
            break;
        case EObjReco_PoseNet:
            break;
        default:
            break;
    }
    return nullptr;
}

void CObjReco::Destroy(CObjReco * pHandle)
{
    delete pHandle;
}
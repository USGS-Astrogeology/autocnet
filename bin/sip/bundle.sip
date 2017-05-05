// SIP Wrapper to the ISIS3 bundle class
%Module bundle

%Import QtCore/QtCoremod.sip
%Import QtXml/QtXmlmod.sip

namespace Isis
{

    class BundleAdjust {
      %TypeHeaderCode
      #include "BundleAdjust.h"
      %End
    public:
    BundleAdjust(const QString &cnetFile,
                 const QString &cubeList, bool printSummary = true);
    };
};

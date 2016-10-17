// SIP Wrapper to the ISIS3 bundle class

%Import QtCore/QtCoremod.sip
%Import QtXml/QtXmlmod.sip

%Include type_conversions.sip

namespace Isis {
  class BundleAdjust : public QObject {
    %TypeHeaderCode
    #include "BundleAdjust.h"
    %End

  public:
    BundleAdjust(QSharedPointer<Isis::BundleSettings> settings, const QString &cnetFile, const QString &cubeList, bool printSummary /In/ = true);
    Isis::BundleSolutionInfo solveCholeskyBR();
   };
};

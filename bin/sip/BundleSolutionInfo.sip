
%Include type_conversions.sip

namespace Isis{

class BundleSolutionInfo : public QObject {
  %TypeHeaderCode
    #include "BundleSolutionInfo.h"
  %End

 public:
   BundleSolutionInfo(const Isis::BundleSolutionInfo &src);

   void setOutputStatistics(Isis::BundleResults statisticsResults);
   void setRunTime(QString runTime);

   QString id() const;
   QString controlNetworkFileName() const;
   QSharedPointer<Isis::BundleSettings> bundleSettings();
   Isis::BundleResults bundleResults();
   QString runTime() const;

   bool outputText();
   bool outputPointsCSV();
   bool outputResiduals();
}; // end namespace Isis
};

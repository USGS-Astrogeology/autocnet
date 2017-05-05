namespace Isis {

  class ControlNet : public QObject {
    %TypeHeaderCode
    #include "ControlNet.h"
    %End
    public:
      ControlNet();
      ControlNet(const Isis::ControlNet &other);

      void clear();

      void Write(const QString &filename, bool pvl = false);

      bool IsValid() const;
      double GetMaximumResidual();
      QString GetNetworkId() const;
      int GetNumEditLockMeasures();
      int GetNumEditLockPoints();
      int GetNumIgnoredMeasures();
      int GetNumberOfValidMeasuresInImage(const QString &serialNumber);
      int GetNumberOfJigsawRejectedMeasuresInImage(const QString &serialNumber);
      void ClearJigsawRejected();
      void IncrementNumberOfRejectedMeasuresInImage(const QString &serialNumber);
      void DecrementNumberOfRejectedMeasuresInImage(const QString &serialNumber);
      int GetNumMeasures() const;
      int GetNumPoints() const;
      int GetNumValidMeasures();
      int GetNumValidPoints();
      QString GetTarget() const;
      QString GetUserName() const;
    };
};

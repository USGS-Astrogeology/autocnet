%Include Statistics.sip
%Include type_conversions.sip


namespace Isis {
  class BundleResults : public QObject {
    %TypeHeaderCode
      #include "BundleResults.h"
    %End

    public:
      BundleResults(const Isis::BundleResults &src);
      void initialize();

      void resizeSigmaStatisticsVectors(int numberImages);
      // void setRmsImageResidualLists(QList<Isis::Statistics> rmsImageLineResiduals,
      //                               QList<Isis::Statistics> rmsImageSampleResiduals,
      //                               QList<Isis::Statistics> rmsImageResiduals);
      // void setRmsImageResidualLists(QVector<Isis::Statistics> rmsImageLineResiduals,
      //                               QVector<Isis::Statistics> rmsImageSampleResiduals,
      //                               QVector<Isis::Statistics> rmsImageResiduals);
      // void setSigmaLatitudeRange(Distance minLatDist, Distance maxLatDist,
      //                            QString minLatPointId, QString maxLatPointId);
      // void setSigmaLongitudeRange(Distance minLonDist, Distance maxLonDist,
      //                             QString minLonPointId, QString maxLonPointId);
      // void setSigmaRadiusRange(Distance minRadDist, Distance maxRadDist,
      //                          QString minRadPointId, QString maxRadPointId);
      void setRmsFromSigmaStatistics(double rmsFromSigmaLatStats,
                                     double rmsFromSigmaLonStats,
                                     double rmsFromSigmaRadStats);

      void printMaximumLikelihoodTierInformation();
      void incrementMaximumLikelihoodModelIndex();

      void incrementFixedPoints();
      int numberFixedPoints() const;
      void incrementHeldImages();
      int numberHeldImages() const;
      void incrementIgnoredPoints();
      int numberIgnoredPoints() const; // currently unused ???
      void setRejectionLimit(double rejectionLimit);
      void setRmsXYResiduals(double rx, double ry, double rxy);

      void setNumberRejectedObservations(int numberObservations);
      void setNumberObservations(int numberObservations);
      void setNumberImageParameters(int numberParameters); // ??? this is the same value an m_nRank
      void resetNumberConstrainedPointParameters();
      void incrementNumberConstrainedPointParameters(int incrementAmount);
      void resetNumberConstrainedImageParameters();
      void incrementNumberConstrainedImageParameters(int incrementAmount);
      void resetNumberConstrainedTargetParameters();
      void incrementNumberConstrainedTargetParameters(int incrementAmount);
      void setNumberUnknownParameters(int numberParameters);
      void computeDegreesOfFreedom();

      void computeSigma0(double dvtpv, Isis::BundleSettings::ConvergenceCriteria criteria);
      void setDegreesOfFreedom(double degreesOfFreedom);
      void setSigma0(double sigma0);
      void setElapsedTime(double time);
      void setElapsedTimeErrorProp(double time);
      void setRadiansToMeters(double rtm);
      void setConverged(bool converged); // or initialze method
      // void setBundleControlPoints(QVector<BundleControlPointQsp> controlPoints);
      // void setOutputControlNet(QSharedPointer<Isis::ControlNet> outNet);
      void setIterations(int iterations);
      // void setObservations(BundleObservationVector observations);

      QList<Isis::Statistics> rmsImageSampleResiduals() const;
      QList<Isis::Statistics> rmsImageLineResiduals() const;
      QList<Isis::Statistics> rmsImageResiduals() const;
      QVector<Isis::Statistics> rmsImageXSigmas() const;       // currently unused ???
      QVector<Isis::Statistics> rmsImageYSigmas() const;       // currently unused ???
      QVector<Isis::Statistics> rmsImageZSigmas() const;       // currently unused ???
      QVector<Isis::Statistics> rmsImageRASigmas() const;      // currently unused ???
      QVector<Isis::Statistics> rmsImageDECSigmas() const;     // currently unused ???
      QVector<Isis::Statistics> rmsImageTWISTSigmas() const;   // currently unused ???

      // Distance minSigmaLatitudeDistance() const;
      // Distance maxSigmaLatitudeDistance() const;
      // Distance minSigmaLongitudeDistance() const;
      // Distance maxSigmaLongitudeDistance() const;
      // Distance minSigmaRadiusDistance() const;
      // Distance maxSigmaRadiusDistance() const;

      QString maxSigmaLatitudePointId() const;
      QString minSigmaLatitudePointId() const;
      QString minSigmaLongitudePointId() const;
      QString maxSigmaLongitudePointId() const;
      QString minSigmaRadiusPointId() const;
      QString maxSigmaRadiusPointId() const;
      double sigmaLatitudeStatisticsRms() const;
      double sigmaLongitudeStatisticsRms() const;
      double sigmaRadiusStatisticsRms() const;
      double rmsRx() const;  // currently unused ???
      double rmsRy() const;  // currently unused ???
      double rmsRxy() const; // currently unused ???
      double rejectionLimit() const;
      double radiansToMeters() const;
      int numberRejectedObservations() const;
      int numberObservations() const;

      int numberImageParameters() const; // ??? this is the same value an m_nRank
      int numberConstrainedPointParameters() const;
      int numberConstrainedImageParameters() const;
      int numberConstrainedTargetParameters() const;
      int numberUnknownParameters() const;
      int degreesOfFreedom() const;
      double sigma0() const;
      double elapsedTime() const;
      double elapsedTimeErrorProp() const;
      bool converged() const; // or initialze method
      // QVector<BundleControlPointQsp> &bundleControlPoints();
      QSharedPointer<Isis::ControlNet> outputControlNet() const;
      int iterations() const;

      int numberMaximumLikelihoodModels() const;
      int maximumLikelihoodModelIndex() const;
      double maximumLikelihoodMedianR2Residuals() const;
      double maximumLikelihoodModelQuantile(int modelIndex) const;
    };

};

%Include BundleObservationSolveSettings.sip
%Include MaximumLikelihoodWFunctions.sip

namespace Isis {

  class BundleSettings  {
    %TypeHeaderCode
    #include "BundleSettings.h"
    %End
  public:

    enum ConvergenceCriteria {
      Sigma0,              /**< The value of sigma0 will be used to determine that the bundle
                                adjustment has converged.*/
      ParameterCorrections /**< All parameter corrections will be used to determine that the
                                bundle adjustment has converged.*/
    };

    // enum MaximumLikelihoodModel {
    //   NoMaximumLikelihoodEstimator,
    //   Huber,
    //   ModifiedHuber,
    //   Welsch,
    //   Chen
    // };

    BundleSettings();
    void setValidateNetwork(bool);

    void setSolveOptions(bool solveObservationMode = false,
                         bool updateCubeLabel = false,
                         bool errorPropagation = false,
                         bool solveRadius = false,
                         double globalLatitudeAprioriSigma = Isis::Null,
                         double globalLongitudeAprioriSigma = Isis::Null,
                         double globalRadiusAprioriSigma = Isis::Null);

   void setOutlierRejection(bool outlierRejection,
                            double multiplier = 1.0);

   bool solveObservationMode() const;
   bool solveRadius() const;
   bool updateCubeLabel() const;
   bool errorPropagation() const;
   bool outlierRejection() const;
   double outlierRejectionMultiplier() const;
   double globalLatitudeAprioriSigma() const;
   double globalLongitudeAprioriSigma() const;
   double globalRadiusAprioriSigma() const;

   int numberSolveSettings() const;
   Isis::BundleObservationSolveSettings observationSolveSettings(QString instrumentId) const;
   Isis::BundleObservationSolveSettings observationSolveSettings(int n) const;

   SIP_PYLIST maximumLikelihoodEstimatorModels() const;
   %MethodCode // convert QList<QPair> to Python list if tuples
     typedef Isis::MaximumLikelihoodWFunctions::Model Model;
     size_t size = sipCpp->maximumLikelihoodEstimatorModels().size();
     PyObject *l = PyList_New(size);
     for (size_t i = 0; i < size; ++i){
       // get the two items from the QPair
       Model *model = new Model(sipCpp->maximumLikelihoodEstimatorModels()[i].first);
       double doubl = sipCpp->maximumLikelihoodEstimatorModels()[i].second;

       // Convert the two items to PyObjects
       PyObject *first = sipConvertFromType((void*)(model), sipType_Isis_MaximumLikelihoodWFunctions_Model, NULL);
       PyObject *second = PyFloat_FromDouble(doubl);

       // set the tuple items
       PyObject *tuple = PyTuple_New(2);
       PyTuple_SetItem(tuple, 0, first);
       PyTuple_SetItem(tuple, 1, second);

       // Add new tuple to list
       PyList_SetItem(l, i, tuple);
     }
     return l;
   %End


   int numberTargetBodyParameters() const;
   bool solveTargetBody() const;
   bool solvePoleRA() const;
   bool solvePoleRAVelocity() const;
   bool solvePoleDec() const;
   bool solvePoleDecVelocity() const;
   bool solvePM() const;
   bool solvePMVelocity() const;
   bool solvePMAcceleration() const;
   bool solveTriaxialRadii() const;
   bool solveMeanRadius() const;

  };
};

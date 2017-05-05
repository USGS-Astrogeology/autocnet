namespace Isis {

  class MaximumLikelihoodWFunctions {
    %TypeHeaderCode
    #include "MaximumLikelihoodWFunctions.h"
    %End

  public:

    enum Model {
      Huber,
      HuberModified,
      Welsch,
      Chen
    };

    static QString modelToString(Model model);
    static Isis::MaximumLikelihoodWFunctions::Model stringToModel(QString modelName);

    MaximumLikelihoodWFunctions();
    MaximumLikelihoodWFunctions(Model modelSelection);
    MaximumLikelihoodWFunctions(Model modelSelection, double tweakingConstant);
    MaximumLikelihoodWFunctions(const Isis::MaximumLikelihoodWFunctions &other);

    void setModel(Model modelSelection); // uses default tweaking constant
    void setTweakingConstantDefault();

    void setModel(Model modelSelection, double tweakingConstant);
    void setTweakingConstant(double tweakingConstant);

    Model model() const;
    double tweakingConstant() const;

    // the W functions provide an additional weighting factor W which is used
    // to 're-weight' each observation dynamically during an adjustment, the
    // scalar functions provide access to various flavors of this scalar (as
    // a function of the residual divided by the residuals sigma)

    double sqrtWeightScaler(double residualZScore); //it is often convient to use square roots of
                                                    //weights when building normals, this function
                                                    // provides the scaler for the square root of
                                                    // the weight directly
    double tweakingConstantQuantile(); // returns which quantile of the residuals is recommended to
                                       // use as the tweaking constant, this varies as a function of
                                       // the model being employed

    QString weightedResidualCutoff();

    QDataStream &write(QDataStream &stream) const;
    QDataStream &read(QDataStream &stream);
  };
};

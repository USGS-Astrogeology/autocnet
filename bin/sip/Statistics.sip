
namespace Isis {
  class Statistics : public QObject {
    %TypeHeaderCode
      #include "Statistics.h"
    %End
    public:
      Statistics(const Isis::Statistics &other);

      void Reset();

      void AddData(const double *data, const unsigned int count);
      void AddData(const double data);

      void RemoveData(const double *data, const unsigned int count);
      void RemoveData(const double data);

      double ValidMinimum() const;
      double ValidMaximum() const;
      bool InRange(const double value);
      bool AboveRange(const double value);
      bool BelowRange(const double value);

      double Average() const;
      double StandardDeviation() const;
      double Variance() const;
      double Sum() const;
      double SumSquare() const;
      double Rms() const;

      double Minimum() const;
      double Maximum() const;
      double ChebyshevMinimum(const double percent = 99.5) const;
      double ChebyshevMaximum(const double percent = 99.5) const;
      double BestMinimum(const double percent = 99.5) const;
      double BestMaximum(const double percent = 99.5) const;
      double ZScore(const double value) const;

      bool RemovedData() const;
    };
};

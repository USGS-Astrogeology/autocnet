
%MappedType QList<Isis::Statistics>
{
  %TypeHeaderCode
  #include "Statistics.h"
  #include <QList>
  %End

  %ConvertFromTypeCode
    size_t size = sipCpp->size();
    PyObject *l = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
      Isis::Statistics* cppStats = new Isis::Statistics(sipCpp->at(i));
      PyObject *pyStats = sipConvertFromType((void*)(cppStats), sipType_Isis_Statistics, NULL);
      PyList_SetItem(l, i, pyStats);
    }

    return l;
  %End


  %ConvertToTypeCode
    Py_ssize_t size = PyList_Size(sipPy);
    QList<Isis::Statistics> *cppStatList = new QList<Isis::Statistics>();
    int sipErr = 0;
    for(int i = 0; i < size; i++) {
      PyObject *pyStat = PyList_GET_ITEM(sipPy, i);
      Isis::Statistics *stats = (Isis::Statistics*)sipConvertToType(pyStat, sipType_Isis_Statistics, NULL, SIP_NOT_NONE, NULL, &sipErr);
      cppStatList->append(*stats);
    }

    *sipCppPtr = cppStatList;
    return sipGetState(sipTransferObj);
  %End
};



// convert Bundlesettings wrapped in a QSharedPointer in Isis
%MappedType QSharedPointer<Isis::BundleSettings>
{
  %TypeHeaderCode
  #include "BundleSettings.h"
  #include <QSharedPointer>
  using namespace std;
  %End

  %ConvertFromTypeCode
    Isis::BundleSettings *settings = sipCpp->data();
    PyObject *pySettings = sipConvertFromType(settings, sipType_Isis_BundleSettings, NULL);
    return pySettings;
  %End

  %ConvertToTypeCode
    int sipErr = 0;
    QSharedPointer<Isis::BundleSettings> *wrappedSettings;
    if (!sipCanConvertToType(sipPy, sipType_Isis_BundleSettings, SIP_NOT_NONE)) {
        sipErr = 1;
    }
    else {
        Isis::BundleSettings *settings = (BundleSettings*) sipConvertToType(sipPy, sipType_Isis_BundleSettings, NULL, SIP_NOT_NONE, NULL, &sipErr);
        wrappedSettings->reset(settings);
    }

    *sipCppPtr = wrappedSettings;
    return sipGetState(sipTransferObj);
  %End
};


// convert Bundlesettings wrapped in a QSharedPointer in Isis
%MappedType QSharedPointer<Isis::ControlNet>
{
  %TypeHeaderCode
  #include "ControlNet.h"
  #include <QSharedPointer>
  using namespace std;
  %End

  %ConvertFromTypeCode
    PyObject *pySettings = sipConvertFromType(sipCpp, sipType_Isis_ControlNet, NULL);
    return pySettings;
  %End

  %ConvertToTypeCode
    int sipErr = 0;
    QSharedPointer<Isis::ControlNet> *qspControlNet;
    if (!sipCanConvertToType(sipPy, sipType_Isis_ControlNet, SIP_NOT_NONE)) {
        sipErr = 1;
    }
    else {
        Isis::ControlNet *cppControlNet = (Isis::ControlNet*)sipConvertToType(sipPy, sipType_Isis_ControlNet, NULL, SIP_NOT_NONE, NULL, &sipErr);
        qspControlNet->reset(cppControlNet);
    }

    *sipCppPtr = qspControlNet;
    return sipGetState(sipTransferObj);
  %End
};



// Convert a python str object to a std::string.
%MappedType std::string
{
  %TypeHeaderCode
  #include <string>
  %End

  %ConvertFromTypeCode
      // convert an std::string to a Python (unicode) string
      PyObject* newstring;
      newstring = PyUnicode_DecodeUTF8(sipCpp->c_str(), sipCpp->length(), NULL);
      if(newstring == NULL) {
          PyErr_Clear();
          newstring = PyBytes_FromString(sipCpp->c_str());
      }
      return newstring;
  %End

  %ConvertToTypeCode
      // Allow a Python string (or a unicode string) whenever a string is
      // expected.
      // If argument is a Unicode string, just decode it to UTF-8
      // If argument is a Python string, assume it's UTF-8
      if (sipIsErr == NULL)
          return (PyBytes_Check(sipPy) || PyUnicode_Check(sipPy));

      if (sipPy == Py_None)
      {
          *sipCppPtr = new std::string;
          return 1;
      }

      if (PyUnicode_Check(sipPy))
      {
          PyObject* s = PyUnicode_AsEncodedString(sipPy, "UTF-8", "");
          *sipCppPtr = new std::string(PyBytes_AS_STRING(s));
          Py_DECREF(s);
          return 1;
      }

      if (PyBytes_Check(sipPy))
      {
          *sipCppPtr = new std::string(PyBytes_AS_STRING(sipPy));
          return 1;
      }
      return 0;
  %End
};

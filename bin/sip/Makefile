include /work/users/krodriguez/autocnet/isis/make/isismake.os

TARGET = bundle.so
OFILES = sipbundlecmodule.o sipbundleQVector0100IsisStatistics.o sipbundleIsisBundleAdjust.o sipbundleIsisBundleSettings.o sipbundleIsisMaximumLikelihoodWFunctions.o sipbundleIsisBundleObservationSolveSettings.o sipbundleIsisBundleSolutionInfo.o sipbundleIsisBundleResults.o sipbundleIsisStatistics.o sipbundleIsisControlNet.o sipbundleIsis.o sipbundlestdstring.o sipbundleQSharedPointer0100IsisControlNet.o sipbundleQSharedPointer0100IsisBundleSettings.o sipbundleQList0100IsisStatistics.o
HFILES = sipAPIbundle.h 

CC = gcc
CXX = g++
LINK = g++
CPPFLAGS = -DNDEBUG -I. -I/home/krodriguez/miniconda3/envs/linux_x64/include/python3.5m
CFLAGS = -pipe -fPIC -O2 -Wall -W
CXXFLAGS = $(ALLINCDIRS) -pipe -fPIC -O2 -Wall -W
LFLAGS = $(ALLLIBDIRS) -shared -Wl,--version-script=bundle.exp
LIBS = $(ALLLIBS) 
.SUFFIXES: .c .o .cpp .cc .cxx .C


.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.C.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

all: $(TARGET)

$(OFILES): $(HFILES)

$(TARGET): $(OFILES)
	@echo '{ global: PyInit_bundle; local: *; };' > bundle.exp
	$(LINK) $(LFLAGS) -o $(TARGET) $(OFILES) $(LIBS)

install: $(TARGET)
	@test -d $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/lib/python3.5/site-packages || mkdir -p $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/lib/python3.5/site-packages
	cp -f $(TARGET) $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/lib/python3.5/site-packages/$(TARGET)
	strip $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/lib/python3.5/site-packages/$(TARGET)
	@test -d $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/share/sip/PyQt5/isis3 || mkdir -p $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/share/sip/PyQt5/isis3
	cp -f master.sip $(DESTDIR)/home/krodriguez/miniconda3/envs/linux_x64/share/sip/PyQt5/isis3/master.sip

clean:
	-rm -f $(TARGET)
	-rm -f sipbundlecmodule.o
	-rm -f sipbundleQVector0100IsisStatistics.o
	-rm -f sipbundleIsisBundleAdjust.o
	-rm -f sipbundleIsisBundleSettings.o
	-rm -f sipbundleIsisMaximumLikelihoodWFunctions.o
	-rm -f sipbundleIsisBundleObservationSolveSettings.o
	-rm -f sipbundleIsisBundleSolutionInfo.o
	-rm -f sipbundleIsisBundleResults.o
	-rm -f sipbundleIsisStatistics.o
	-rm -f sipbundleIsisControlNet.o
	-rm -f sipbundleIsis.o
	-rm -f sipbundlestdstring.o
	-rm -f sipbundleQSharedPointer0100IsisControlNet.o
	-rm -f sipbundleQSharedPointer0100IsisBundleSettings.o
	-rm -f sipbundleQList0100IsisStatistics.o
	-rm -f bundle.exp

#include <Python.h>
#define NO_IMPORT_ARRAY
#if PY_MAJOR_VERSION >= 3
#define NPY_NO_DEPRECATED_API 0x0
#endif
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>
#include <stdio.h>  
#include <sys/types.h>
#include <new>
#include <complex>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION



// cribbed from SWIG machinery
#if PY_MAJOR_VERSION >= 3
#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
//#define PyString_AsString(str) PyBytes_AsString(str)
#define PyString_Size(str) PyBytes_Size(str)
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define PyString_AS_STRING(x) PyUnicode_AS_STRING(x)
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)
#endif
// and after some hacking
#if PY_MAJOR_VERSION >= 3
#define PyString_AsString(obj) PyUnicode_AsUTF8(obj)
#endif







/* Docstrings */
static char module_docstring[] =
    "Solver of leakage terms and source polarization.";
static char setData_docstring[] =
    "Reads data pointers and arranges data.";
static char getHessian_docstring[] =
    "Computes the Hessian matrix and residuals vector, given the parameter values";



/* Available functions */
static PyObject *setData(PyObject *self, PyObject *args);
static PyObject *getHessian(PyObject *self, PyObject *args);



/* Module specification */
static PyMethodDef module_methods[] = {
    {"setData", setData, METH_VARARGS, setData_docstring},
    {"getHessian", getHessian, METH_VARARGS, getHessian_docstring},
    {NULL, NULL, 0, NULL}
};


/* Initialize the module */


/* Initialize the module */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pc_module_def = {
    PyModuleDef_HEAD_INIT,
    "PolSolver",          /* m_name */
    module_docstring,       /* m_doc */
    -1,                     /* m_size */
    module_methods,         /* m_methods */
    NULL,NULL,NULL,NULL     /* m_reload, m_traverse, m_clear, m_free */
};

PyMODINIT_FUNC PyInit_PolSolver(void)
{
    PyObject *m = PyModule_Create(&pc_module_def);
    return(m);
}


#else

PyMODINIT_FUNC initPolSolver(void)
{
    PyObject *m = Py_InitModule3("PolSolver", module_methods, module_docstring);
    if (m == NULL)
        return;

/* Load `numpy` functionality. */
    //import_array();


}

#endif







// Global variables:
static bool DEBUG = false;
bool doCirc, doOrd1, FitPerIF, doPfrac, isMod, doFaraday;  
typedef std::complex<double> cplx64d;

int Nchan, NCalSour, Nterms, Nspw; 
int *Nvis, *NSou; 
int NAnt, NPar;
int NITER = 0;
int **A1, **A2, **SpwId;
double *VARS, *Hessian, *DerVec;
double ***Wgt; // = new double**[1];
double **WgtCorr, **NuPow, **PAng1, **PAng2;
double MaxPfrac;
double *TotFlux, **Lambdas, **LogLambdas;
cplx64d **DATA, **COMPS;
cplx64d *DR, *DL, **EPA, **EMA, ***DRp, ***DLp;

/*
TotFlux = new double[1];
NSou = new int[1];
Nvis = new int[1];
DATA = new cplx64d*[1];
COMPS = new cplx64d*[1];
EPA = new cplx64d*[1];
EMA = new cplx64d*[1];
A1 = new int*[1];
A2 = new int*[1];
Wgt = new double**[1];
Wgt[0] = new double*[4];
*/

int *PSou, *PAntR, *PAntL, *VSou, *VAntR, *VAntL;






//////////////////////////////////
// Gets the pointers to data, metadata, Hessian and residuals vector:
static PyObject *setData(PyObject *self, PyObject *args)
{

  PyObject *DATAPy, *A1Py, *A2Py, *COMPSPy, *EPAsPy, *EMAsPy, *PSouPy, *VARSPy, *FluxPy, *NuPowPy, *SpwIdPy, *MODPy, *LambdasPy; 
  PyObject *PAntRPy, *PAntLPy, *WgtPy, *WgtCorrPy, *VSouPy, *VAntRPy, *VAntLPy, *HessianPy, *ResVecPy, *PAng1Py, *PAng2Py;

  PyObject *Err;
  int i,j;


  if (!PyArg_ParseTuple(args,"OOOOOOOOOOOOOOOOOOOOOObbbiOOdb", &DATAPy, &WgtPy, &WgtCorrPy, &NuPowPy, &LambdasPy, &SpwIdPy, &A1Py, &A2Py, &COMPSPy, &MODPy, &EPAsPy, &EMAsPy, &PSouPy, &PAntRPy, &PAntLPy, &VSouPy, &VAntRPy, &VAntLPy, &VARSPy, &HessianPy, &ResVecPy,&FluxPy, &doCirc, &doOrd1, &FitPerIF, &Nspw, &PAng1Py, &PAng2Py, &MaxPfrac,&doFaraday))
    {printf("Failed setData! Wrong arguments!\n"); fflush(stdout); Err = Py_BuildValue("i",-1); return Err;};

//  delete NSou, Nvis; 
//  delete[] DATA, COMPS, EPA, EMA, A1, A2, Wgt, DR, DL;
  NCalSour = PyList_Size(DATAPy);
  Nterms = PyList_Size(NuPowPy);

  doPfrac = (MaxPfrac>0.0);

  NuPow = new double*[Nterms];
  NSou = new int[NCalSour];
  Nvis = new int[NCalSour];
  DATA = new cplx64d*[NCalSour];
  COMPS = new cplx64d*[NCalSour];
  EPA = new cplx64d*[NCalSour];
  EMA = new cplx64d*[NCalSour];
  A1 = new int*[NCalSour];
  A2 = new int*[NCalSour];
  PAng1 = new double*[NCalSour];
  PAng2 = new double*[NCalSour];
  SpwId = new int*[NCalSour];
  Wgt = new double**[NCalSour];
  WgtCorr = new double *[NCalSour];

  Hessian = (double *)PyArray_DATA(HessianPy);
  DerVec = (double *)PyArray_DATA(ResVecPy);
  NPar =  PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(ResVecPy))[0];

  TotFlux = (double *)PyArray_DATA(FluxPy);

  if(DEBUG){printf("setData variables read. %i sources; %i parameters\n",NCalSour,NPar); fflush(stdout);};

// get pointers to the data:


// Array must be of proper full-polarization data:
  int Ndims = PyArray_NDIM(PyList_GetItem(DATAPy,0));
  if (Ndims != 3){printf("Array must have three dimenstions!\n"); fflush(stdout); Err = Py_BuildValue("i",-2); return Err;};
  NAnt = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(PAntRPy))[0];
  Nchan = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(PyList_GetItem(DATAPy,0)))[1];

  if(DEBUG){printf("%i antennas; %i frequency channels\n",NAnt,Nchan); fflush(stdout);};



  Lambdas = new double*[Nspw];
  LogLambdas = new double*[Nspw];



  for(i=0; i<Nspw; i++){
    Lambdas[i] = (double *)PyArray_DATA(PyList_GetItem(LambdasPy,i));
    LogLambdas[i] = new double[Nchan];
    for(j=0; j<Nchan;j++){
      if (Lambdas[i][j] > 0.0){LogLambdas[i][j] = std::log(Lambdas[0][0]/Lambdas[i][j]);} else {LogLambdas[i][j]=0.0;};
    };
  };


  DRp = new cplx64d**[NAnt];
  DLp = new cplx64d**[NAnt];

  for(i=0;i<NAnt;i++){
    DRp[i] = new cplx64d*[Nspw];
    DLp[i] = new cplx64d*[Nspw];
    for(j=0;j<Nspw;j++){
      DRp[i][j] = new cplx64d[Nchan];
      DLp[i][j] = new cplx64d[Nchan];
    };
  };


  for(i=0;i<Nterms;i++){
    NuPow[i] = (double *)PyArray_DATA(PyList_GetItem(NuPowPy,i));
  };

  isMod =  PyList_Size(MODPy)>PyList_Size(COMPSPy);

  for(i=0;i<NCalSour;i++){

    if(DEBUG){printf("Setting data for field %i:\n",i); fflush(stdout);};

    Nvis[i] = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(PyList_GetItem(DATAPy,i)))[0];

    if (isMod){
      NSou[i] = 0;
    } else {
      NSou[i] = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(PyList_GetItem(COMPSPy,i)))[2] - 1;
    };

    if(DEBUG){printf("   %i visibilities; %i subcomponents. \n",Nvis[i],NSou[i]); fflush(stdout);};


    DATA[i] = (cplx64d *)PyArray_DATA(PyList_GetItem(DATAPy,i));
    A1[i] = (int *)PyArray_DATA(PyList_GetItem(A1Py,i));
    A2[i] = (int *)PyArray_DATA(PyList_GetItem(A2Py,i));
    SpwId[i] = (int *)PyArray_DATA(PyList_GetItem(SpwIdPy,i));

    if (isMod){
      COMPS[i] = (cplx64d *)PyArray_DATA(PyList_GetItem(MODPy,i));
    } else {
      COMPS[i] = (cplx64d *)PyArray_DATA(PyList_GetItem(COMPSPy,i));
    };

    EPA[i] = (cplx64d *)PyArray_DATA(PyList_GetItem(EPAsPy,i));
    EMA[i] = (cplx64d *)PyArray_DATA(PyList_GetItem(EMAsPy,i));

    PAng1[i] = (double *)PyArray_DATA(PyList_GetItem(PAng1Py,i));
    PAng2[i] = (double *)PyArray_DATA(PyList_GetItem(PAng2Py,i));

    if(DEBUG){printf("Setting weights for field %i\n",i); fflush(stdout);};
    Wgt[i] = new double*[4];
    for(j=0;j<4;j++){
      if(DEBUG){printf("PolProduct %i ",j); fflush(stdout);};
      Wgt[i][j] = (double *)PyArray_DATA(PyList_GetItem(PyList_GetItem(WgtPy,i),j));
      if(DEBUG){printf("Done.\n"); fflush(stdout);};

    };

    WgtCorr[i] = (double *)PyArray_DATA(PyList_GetItem(WgtCorrPy,i));

  };


  if(DEBUG){printf("Getting pointers to parameter indices\n"); fflush(stdout);};

  PSou = (int *)PyArray_DATA(PSouPy);
  PAntR = (int *)PyArray_DATA(PAntRPy);
  PAntL = (int *)PyArray_DATA(PAntLPy);
  VSou = (int *) PyArray_DATA(VSouPy);
  VAntR = (int *)PyArray_DATA(VAntRPy);
  VAntL = (int *)PyArray_DATA(VAntLPy);
  VARS = (double *)PyArray_DATA(VARSPy);


  if(DEBUG){printf("Array pointers set.\n"); fflush(stdout);};


  DR = reinterpret_cast<cplx64d*>(&VARS[VAntR[0]]);
  DL = reinterpret_cast<cplx64d*>(&VARS[VAntL[0]]);

  if(DEBUG){printf("Dterm pointers re-casted and set.\n"); fflush(stdout);};
  

//  DR = new cplx64d[NAnt]; //reinterpret_cast<cplx64d*>(&VARS[VAntR[0]]);
//  DL = new cplx64d[NAnt]; //reinterpret_cast<cplx64d*>(&VARS[VAntL[0]]);
//  for (i=0;i<NAnt;i++){
//    DR[i] = reinterpret_cast<cplx64d>(VARS[VAntR[i]]);
//    DL[i] = reinterpret_cast<cplx64d>(VARS[VAntL[i]]);
//  };

if (DEBUG){
  int k=0;
  for (j=0; j<NCalSour;j++){
    printf("Field id %i has %i visibs with %i channels\n",j,Nvis[j],Nchan);

    printf("Source subcomponents: \n");
    for (i=0; i< NSou[j]; i++){
      printf("#%i / %i:  PSou = %i,  VSou = %i\n",i,k,PSou[k],VSou[k]);
      printf("  Starting values: %.3e, %.3e\n\n",VARS[VSou[k]],VARS[VSou[k]+1]);
      k+=1;
    };
  };

  printf("ANTENNAS: \n");
  for (i=0; i< NAnt; i++){
    printf("#%i:  PAntR = %i,  VAntR = %i\n",i,PAntR[i],VAntR[i]);
    printf("#%i:  PAntL = %i,  VAntL = %i\n",i,PAntL[i],VAntL[i]);
    printf("  Starting values: %.3e, %.3e | %.3e, %.3e\n\n",DR[i].real(),DR[i].imag(),DL[i].real(),DL[i].imag());
  };

  int chi=Nchan/2;
 // j = 0;
  cplx64d RR, RL, LR, LL;
  double wgtaux;
  for (j=0; j<NCalSour;j++){
  printf("FIRST %i VISIBILITIES (FIELD %i; channel %i): \n",2*NAnt,j,chi);

  for (i=0; i < 2*NAnt; i++){
     RR = DATA[j][chi*4 + i*4*Nchan];
     RL = DATA[j][chi*4 + i*4*Nchan+1];
     LR = DATA[j][chi*4 + i*4*Nchan+2];
     LL = DATA[j][chi*4 + i*4*Nchan+3];
     wgtaux = 0.5*(Wgt[j][0][i*Nchan + chi] + Wgt[j][3][i*Nchan + chi]);
     printf("#%i (wgt %.2e): RR = (%-.2e,%-.2e) | RL = (%-.2e,%-.2e) | LR = (%-.2e,%-.2e) | LL = (%-.2e,%-.2e)\n",i,wgtaux,RR.real(),RR.imag(), RL.real(),RL.imag(), LR.real(),LR.imag(), LL.real(),LL.imag());

    };
  };
};



// Return success:
  NITER = 0;
  Err = Py_BuildValue("i",0);

  return Err;


};





/////////////////////////////////////////////////////////////
// Computes Hessian, residuals vector, and Chi Square.

static PyObject *getHessian(PyObject *self, PyObject *args)
{


  PyObject *Err;

  double Ifac, ChiSq, res, F[4];
  double *IRatio, *Iwt;
  bool doDeriv, doResid, doWgt, auxBool;
  int i,j,k,l, s, term;
  long Ndata;

  IRatio = new double[NCalSour];
  Iwt = new double[NCalSour];


  if(doCirc){ 
      F[0] = 0.; F[1] = 1.; F[2] = 1.; F[3] = 0.;
  } else {
      F[0] = 1.; F[1] = 1.; F[2] = 1.; F[3] = 1.;
  }; // Relative weights for RR, RL, LR, LL / XX, XY, YX, YY
  
  
  double currWgt[4], ord2, Pf2Abs, PfAbs, ItotAbs, LambEV, LambPow;
  double LambCos, LambSin;
  
  // Turn on/off 2nd order corrections:
  if(doOrd1){ord2=0.0;} else {ord2 = 1.0;};
  // FOR TESTING:
  //   ord2 = 0.0;

  int VisPar[NPar];
  int currPar;

  int iaux1, iaux2, iaux3, iaux4,iaux5;
  int printedGood = 0;

  if (!PyArg_ParseTuple(args,"bbb", &doDeriv, &doResid, &doWgt))
    {printf("Failed getHessian! Wrong arguments!\n"); 
     Err = Py_BuildValue("i",-1); fflush(stdout); return Err;};

  if (DEBUG){
     printf("\n Called getHessian. Num. sources: %i, Num. Params: %i\n",NCalSour,NPar);
     Ndata = 0;
     for (s=0; s<NCalSour; s++){
       for (i=0; i<Nvis[s]; i++){
         for(j=0; j<Nchan; j++){
           iaux3 = i*Nchan + j;
           if(Wgt[s][0][iaux3]>0.0 || Wgt[s][3][iaux3]>0.0){Ndata += 1;};
         };
       };
     };
     printf("There are %d good visibilities.\n",Ndata);
  };

  cplx64d Itot, Qtot, Utot, mBreve, mPhas;
  cplx64d RRc, RLc, LRc, LLc;
  cplx64d RRAux, RLAux, LRAux, LLAux;
  cplx64d Im = cplx64d(0.,1.);

  cplx64d DTEff;

  cplx64d RR, RL, LR, LL, auxC1, auxC2;
  cplx64d resid[4];
  cplx64d AllDer[NPar][4];

  ChiSq = 0.0;


// set matrix and vector to zero:
  for(i=0;i<NPar;i++){
    DerVec[i] = 0.;
    for(j=0;j<NPar;j++){
      Hessian[i*NPar + j] = 0.;
    };
  };

  
  Ifac = 0.;
  Ndata = 0;
  



  s=0;
  int NtotSou = 0; // Total number of fittable source subcomponents.
  for(i=0; i<NCalSour; i++){

    IRatio[i] = 1.0; Iwt[i] = 0.0; 
    if(DEBUG){printf("Source %i has %i components.\n",i+1,NSou[i]);};

    for(j=0; j<NSou[i]; j++){
      if(PSou[j+s]>=0){NtotSou += 1;};
    };
    s += NSou[i];
  };


  if(DEBUG){printf("There is a total of %i components to fit.\n",NtotSou);};



// Interpolate Dterms over all frequencies:
if(FitPerIF){
  for (i=0;i<NAnt;i++){
    iaux1 = i*Nspw;
    for (j=0;j<Nspw;j++){
      for (k=0;k<Nchan;k++){
        DRp[i][j][k] = DR[iaux1 + j];
        DLp[i][j][k] = DL[iaux1 + j];
      };
    };
  };
} else {

  for (i=0;i<NAnt;i++){
    iaux1 = i*(Nterms+1);
    for (j=0;j<Nspw;j++){
      for (k=0;k<Nchan;k++){
        DRp[i][j][k] = DR[iaux1];
        DLp[i][j][k] = DL[iaux1];
        for (l=0; l<Nterms;l++){
          DRp[i][j][k] += DR[iaux1 + l+1]*NuPow[l][j + k*Nspw];
          DLp[i][j][k] += DL[iaux1 + l+1]*NuPow[l][j + k*Nspw];
        };
      };
    };
  };

};


if(DEBUG){
  for (i=0;i<NAnt;i++){
    iaux1 = i*Nspw;
    for (j=0;j<Nspw;j++){
      for (k=0;k<Nchan;k++){
        printf("DR (ant %i, Spw %i, Nu %i): %.3e + %.3ej\n",i,j,k,DRp[i][j][k].real(), DRp[i][j][k].imag());
        printf("DL (ant %i, Spw %i, Nu %i): %.3e + %.3ej\n\n",i,j,k,DLp[i][j][k].real(), DLp[i][j][k].imag());
      };
    };
  };

};




// Loop over calibrator sources:
  for (s=0; s<NCalSour; s++){


  int Ns = 0;  // Number of subcomponents already computed.
  int Nf = 0;  // Number of fittable subcomponents already computed.

//////////
  for(i=0; i<s; i++){
    for(j=0; j<NSou[i]; j++){
      if(PSou[j+Ns]>=0){Nf += 1;};
    };
    Ns += NSou[i];
  };
/////////



  auxBool = false;

// Loop over all data in proper order (time, channel, polariz):
  for(i=0; i<Nvis[s]; i++){


    for(j=0; j<Nchan; j++){

// Indices in 1D for the multi-dimensional arrays:
     iaux1 = (i*Nchan + j)*(NSou[s]+1);
     iaux2 = 4*(j + i*Nchan);
     iaux3 = i*Nchan + j;

// Index of current parameter in Hessian matrix:
   //  currPar = 2*Nf;
     currPar = 0;

// Reset vector of derivatives to zero:
     for(k=0;k<NPar;k++){for(iaux4=0;iaux4<4;iaux4++){AllDer[k][iaux4] = cplx64d(0.,0.);};};

// Proceed only if data are good:

     if(Wgt[s][0][iaux3]>0.0 || Wgt[s][3][iaux3]>0.0){

      Ndata += 1;



      if (NSou[s]>0){

//////////////////////
/////  SIMILARITY ASSUMPTION:

      
// Add Fourier transforms of all source components:
      Itot = COMPS[s][iaux1+NSou[s]]/TotFlux[s];
      Qtot = cplx64d(0.,0.);
      Utot = cplx64d(0.,0.);

      for(k=0; k<NSou[s]; k++){
    //    Itot += COMPS[iaux1 + k];
        if (doPfrac){
          if (doFaraday){
             LambPow = std::pow(Lambdas[0][0]/Lambdas[SpwId[s][i]][j],VARS[VSou[k+Ns]+3]); 
             LambEV = (Lambdas[SpwId[s][i]][j]-Lambdas[0][0])*VARS[VSou[k+Ns]+2];
             Ifac = (std::sin(VARS[VSou[k+Ns]])+1.)*MaxPfrac/2.*LambPow;
             Qtot += Ifac*std::cos(2.*(VARS[VSou[k+Ns]+1]+ LambEV ))*COMPS[s][iaux1 + k];
             Utot += Ifac*std::sin(2.*(VARS[VSou[k+Ns]+1]+ LambEV ))*COMPS[s][iaux1 + k];
          } else {
             Ifac = (std::sin(VARS[VSou[k+Ns]])+1.)*MaxPfrac/2.;
             Qtot += Ifac*std::cos(2.*VARS[VSou[k+Ns]+1])*COMPS[s][iaux1 + k];
             Utot += Ifac*std::sin(2.*VARS[VSou[k+Ns]+1])*COMPS[s][iaux1 + k];
          };
        } else {

          if(doFaraday){
            LambCos = std::cos(2.*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0])*VARS[VSou[k+Ns]+2]);
            LambSin = std::sin(2.*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0])*VARS[VSou[k+Ns]+2]);
            LambPow = std::pow(Lambdas[0][0]/Lambdas[SpwId[s][i]][j],VARS[VSou[k+Ns]+3]); 
            Qtot += LambPow*(VARS[VSou[k+Ns]]*LambCos - VARS[VSou[k+Ns]+1]*LambSin)*COMPS[s][iaux1 + k];
            Utot += LambPow*(VARS[VSou[k+Ns]+1]*LambCos + VARS[VSou[k+Ns]]*LambSin)*COMPS[s][iaux1 + k];
          } else {
          Qtot += VARS[VSou[k+Ns]]*COMPS[s][iaux1 + k];
          Utot += VARS[VSou[k+Ns]+1]*COMPS[s][iaux1 + k];
          };

        };


        iaux4 = PSou[k+Ns];

        if(DEBUG && !auxBool){
          if (doPfrac){
            printf("Sou %i, Comp %i: Pfrac = %.4f | EVPA = %.4f | Q = %.4f , U = %.4f \n",s,k,Ifac,VARS[VSou[k+Ns]+1], Ifac*std::cos(2.*VARS[VSou[k+Ns]+1]),Ifac*std::sin(2.*VARS[VSou[k+Ns]+1]));  
          } else {
            printf("Sou %i, Comp %i: Q = %.4f | U = %.4f\n",s,k,VARS[VSou[k+Ns]],VARS[VSou[k+Ns]+1]);  
          };
        };


        if(doDeriv && iaux4>=0){
          VisPar[currPar] = iaux4;
          VisPar[currPar+1] = iaux4+1;
          currPar += 2;


////////// FIT IN EVPA SPACE:
          if (doPfrac){

            RRc = (std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            RLc = (EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            LRc = (1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            LLc = (std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);

            if (doFaraday){

            VisPar[currPar] = iaux4+2;
            VisPar[currPar+1] = iaux4+3;
            currPar += 2;
     
// Derivative w.r.t. fractional polarization:
            auxC1 = std::cos(VARS[VSou[k+Ns]])*(std::cos(2.*(VARS[VSou[k+Ns]+1]+LambEV)))*COMPS[s][iaux1+k]*MaxPfrac/2.*LambPow;
            auxC2 = std::cos(VARS[VSou[k+Ns]])*(Im*std::sin(2.*(VARS[VSou[k+Ns]+1]+LambEV)))*COMPS[s][iaux1+k]*MaxPfrac/2.*LambPow;
            AllDer[iaux4][0] = ord2*(auxC1+auxC2)*RRc; //(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4][1] = (auxC1+auxC2)*RLc; //(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4][2] = (auxC1-auxC2)*LRc; // (1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4][3] = ord2*(auxC1-auxC2)*LLc; //(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
// Derivative w.r.t. EVPA:
            auxC1 = Ifac*(-std::sin(2.*(VARS[VSou[k+Ns]+1]+LambEV)))*COMPS[s][iaux1+k]*2.;
            auxC2 = Ifac*(Im*std::cos(2.*(VARS[VSou[k+Ns]+1]+LambEV)))*COMPS[s][iaux1+k]*2.;
            AllDer[iaux4+1][0] = ord2*(auxC1+auxC2)*RRc; //(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+1][1] = (auxC1+auxC2)*RLc; //(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+1][2] = (auxC1-auxC2)*LRc; //(1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4+1][3] = ord2*(auxC1-auxC2)*LLc; //(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
// Derivative w.r.t. Faraday effects (Rotation Measure):
            AllDer[iaux4+2][0] = AllDer[iaux4+1][0]*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
            AllDer[iaux4+2][1] = AllDer[iaux4+1][1]*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
            AllDer[iaux4+2][2] = AllDer[iaux4+1][2]*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
            AllDer[iaux4+2][3] = AllDer[iaux4+1][3]*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
// Derivative w.r.t. Faraday effects (Pol. index):
            auxC1 = Ifac*(std::cos(2.*(VARS[VSou[k+Ns]+1]+LambEV)))*COMPS[s][iaux1+k]*LogLambdas[SpwId[s][i]][j];
            auxC2 = Ifac*(Im*std::sin(2.*(VARS[VSou[k+Ns]+1]+LambEV)))*COMPS[s][iaux1+k]*LogLambdas[SpwId[s][i]][j];
            AllDer[iaux4+3][0] = ord2*(auxC1+auxC2)*RRc; //(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+3][1] = (auxC1+auxC2)*RLc; //(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+3][2] = (auxC1-auxC2)*LRc; // (1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4+3][3] = ord2*(auxC1-auxC2)*LLc; //(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);

            } else {

// Derivative w.r.t. fractional polarization:
            auxC1 = std::cos(VARS[VSou[k+Ns]])*(std::cos(2.*VARS[VSou[k+Ns]+1]))*COMPS[s][iaux1+k]*MaxPfrac/2.;
            auxC2 = std::cos(VARS[VSou[k+Ns]])*(Im*std::sin(2.*VARS[VSou[k+Ns]+1]))*COMPS[s][iaux1+k]*MaxPfrac/2.;
            AllDer[iaux4][0] = ord2*(auxC1+auxC2)*RRc; //(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4][1] = (auxC1+auxC2)*RLc; //(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4][2] = (auxC1-auxC2)*LRc; //(1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4][3] = ord2*(auxC1-auxC2)*LLc; //*(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
// Derivative w.r.t. EVPA:
            auxC1 = Ifac*(-std::sin(VARS[VSou[k+Ns]+1]*2.))*COMPS[s][iaux1+k]*2.;
            auxC2 = Ifac*(Im*std::cos(VARS[VSou[k+Ns]+1]*2.))*COMPS[s][iaux1+k]*2.;
            AllDer[iaux4+1][0] = ord2*(auxC1+auxC2)*RRc; //(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+1][1] = (auxC1+auxC2)*RLc; //(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+1][2] = (auxC1-auxC2)*LRc; //(1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4+1][3] = ord2*(auxC1-auxC2)*LLc; //*(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);

            };



          } else {

/////// FIT IN QU SPACE:

            if (doFaraday){

            VisPar[currPar] = iaux4+2;
            VisPar[currPar+1] = iaux4+3;
            currPar += 2;

// Common factor of Derivative w.r.t. Q for all corr products (RR, RL, LR, LL):
            RRc = ord2*COMPS[s][iaux1+k]*(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            RLc = COMPS[s][iaux1+k]*(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            LRc = COMPS[s][iaux1+k]*(1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            LLc = ord2*COMPS[s][iaux1+k]*(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
// Commin factor of Derivative w.r.t. U:
            RR = ord2*Im*COMPS[s][iaux1+k]*(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] - DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            RL = Im*COMPS[s][iaux1+k]*(EPA[s][i] - ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            LR = Im*COMPS[s][iaux1+k]*(-1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            LL = ord2*Im*COMPS[s][iaux1+k]*(-std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);   


// Total derivatives (accounting for the Faraday effects):
            AllDer[iaux4][0] = (RRc*LambCos + RR*LambSin)*LambPow; 
            AllDer[iaux4][1] = (RLc*LambCos + RL*LambSin)*LambPow; 
            AllDer[iaux4][2] = (LRc*LambCos + LR*LambSin)*LambPow; 
            AllDer[iaux4][3] = (LLc*LambCos + LL*LambSin)*LambPow; 
            AllDer[iaux4+1][0] = (RR*LambCos - RRc*LambSin)*LambPow;
            AllDer[iaux4+1][1] = (RL*LambCos - LRc*LambSin)*LambPow;
            AllDer[iaux4+1][2] = (LR*LambCos - LRc*LambSin)*LambPow;
            AllDer[iaux4+1][3] = (LL*LambCos - LLc*LambSin)*LambPow;  
// Deriv w.r.t. Rotation Measure: 
            AllDer[iaux4+2][0] = 2.*(RRc*(-VARS[VSou[k+Ns]]*LambSin-VARS[VSou[k+Ns]+1]*LambCos) + RR*(-VARS[VSou[k+Ns]+1]*LambSin+VARS[VSou[k+Ns]]*LambCos))*LambPow*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
            AllDer[iaux4+2][1] = 2.*(RLc*(-VARS[VSou[k+Ns]]*LambSin-VARS[VSou[k+Ns]+1]*LambCos) + RL*(-VARS[VSou[k+Ns]+1]*LambSin+VARS[VSou[k+Ns]]*LambCos))*LambPow*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
            AllDer[iaux4+2][2] = 2.*(LRc*(-VARS[VSou[k+Ns]]*LambSin-VARS[VSou[k+Ns]+1]*LambCos) + LR*(-VARS[VSou[k+Ns]+1]*LambSin+VARS[VSou[k+Ns]]*LambCos))*LambPow*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);
            AllDer[iaux4+2][3] = 2.*(LLc*(-VARS[VSou[k+Ns]]*LambSin-VARS[VSou[k+Ns]+1]*LambCos) + LL*(-VARS[VSou[k+Ns]+1]*LambSin+VARS[VSou[k+Ns]]*LambCos))*LambPow*(Lambdas[SpwId[s][i]][j]-Lambdas[0][0]);

            RLAux = LambPow*(VARS[VSou[k+Ns]]*LambCos - VARS[VSou[k+Ns]+1]*LambSin)*LogLambdas[SpwId[s][i]][j];
            LRAux = LambPow*(VARS[VSou[k+Ns]+1]*LambCos + VARS[VSou[k+Ns]]*LambSin)*LogLambdas[SpwId[s][i]][j];

            AllDer[iaux4+3][0] =  0.0; 
            AllDer[iaux4+3][1] = (RLAux*RLc + LRAux*RL); 
            AllDer[iaux4+3][2] = (RLAux*LRc + LRAux*LR); 
            AllDer[iaux4+3][3] = 0.0; 

         //   printf("Faraday in QU: %.4e %.4e\n",AllDer[iaux4+2][1].real(),AllDer[iaux4+2][1].imag());

            } else { // No Faraday effects in QU fit (the easiest one):

// Derivative w.r.t. Q for all corr products (RR, RL, LR, LL):
            AllDer[iaux4][0] = ord2*COMPS[s][iaux1+k]*(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] + DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4][1] = COMPS[s][iaux1+k]*(EPA[s][i] + ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4][2] = COMPS[s][iaux1+k]*(1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4][3] = ord2*COMPS[s][iaux1+k]*(std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
// Derivative w.r.t. U:
            AllDer[iaux4+1][0] = ord2*Im*COMPS[s][iaux1+k]*(std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*EPA[s][i] - DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+1][1] = Im*COMPS[s][iaux1+k]*(EPA[s][i] - ord2*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]/EPA[s][i]);
            AllDer[iaux4+1][2] = Im*COMPS[s][iaux1+k]*(-1./EPA[s][i] + ord2*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);
            AllDer[iaux4+1][3] = ord2*Im*COMPS[s][iaux1+k]*(-std::conj(DLp[A2[s][i]][SpwId[s][i]][j])/EPA[s][i] + DLp[A1[s][i]][SpwId[s][i]][j]*EPA[s][i]);   


            };

          };

        };
      };


// Model of Correlation products (in the antenna frame) with no leakage:
      RRc = Itot*EMA[s][i];
      RLc = (Qtot + Im*Utot)*EPA[s][i];
      LRc = (Qtot - Im*Utot)/EPA[s][i];
      LLc = Itot/EMA[s][i];


    } else {


/////////////// NO FIT TO SOURCE POLARIZATION:


/////////////////////////
//// DTERM SELF-CALIBRATION FROM MODEL SOURCE

      RRc =  COMPS[s][iaux2  ];
      RLc =  COMPS[s][iaux2+1]; 
      LRc =  COMPS[s][iaux2+2];
      LLc =  COMPS[s][iaux2+3];

      Itot = RR+LL;


    };



      Ifac = std::abs(Itot);
      currWgt[0] = Wgt[s][0][iaux3]*Ifac;
      currWgt[1] = Wgt[s][1][iaux3]*Ifac;
      currWgt[2] = Wgt[s][2][iaux3]*Ifac;
      currWgt[3] = Wgt[s][3][iaux3]*Ifac;

      




      





      auxBool = true;


// Source's polarization in Fourier space:
      PfAbs = std::abs(RLc);
      Pf2Abs = std::abs(LRc);

// Apply leakage to model visibilities:
      RR = RRc + ord2*(RLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j]) + LRc*DRp[A1[s][i]][SpwId[s][i]][j] + LLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]);

      LL = LLc + ord2*(LRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j]) + RLc*DLp[A1[s][i]][SpwId[s][i]][j] + RRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]);

      RL = (RLc + RRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j]) + LLc*DRp[A1[s][i]][SpwId[s][i]][j] + ord2*LRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j])*DRp[A1[s][i]][SpwId[s][i]][j]);

      LR = (LRc + RRc*DLp[A1[s][i]][SpwId[s][i]][j] + LLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j]) + ord2*RLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j])*DLp[A1[s][i]][SpwId[s][i]][j]);


// Fill-in the fractional polarizations in Fourier space
// (it may be used by the main program to reweight data in a second fit):
      if (doWgt && j==0 && PfAbs>0.0 && Pf2Abs > 0.0 && Ifac > 0.0){
        WgtCorr[s][i] = std::max(PfAbs/Ifac,Pf2Abs/Ifac);
      };



      if(DEBUG && NITER<2 && s==0 && printedGood<5 && j==Nchan/2){
        printf("\n ANTENNA FRAME: \n");
        printf("Model Vis %i Noleak (ANTS %i-%i; FEED ANG.: %.3f,%.3f): RR = (%.2e, %.2e); RL = (%.2e, %.2e); LR = (%.2e, %.2e); LL = (%.2e, %.2e)\n",i,A1[s][i],A2[s][i],PAng1[s][i],PAng2[s][i],RRc.real(),RRc.imag(),RLc.real(),RLc.imag(),LRc.real(),LRc.imag(),LLc.real(),LLc.imag());
        printf("Model Leak   (ANTS %i-%i; FEED ANG.: %.3f,%.3f): RR = (%.2e, %.2e); RL = (%.2e, %.2e); LR = (%.2e, %.2e); LL = (%.2e, %.2e)\n",A1[s][i],A2[s][i],PAng1[s][i],PAng2[s][i],RR.real(),RR.imag(),RL.real(),RL.imag(),LR.real(),LR.imag(),LL.real(),LL.imag());

        printf("\n SKY FRAME: \n");
        RRAux = RRc/EMA[s][i];
        LLAux = LLc*EMA[s][i];
        RLAux = RLc/EPA[s][i];
        LRAux = LRc*EPA[s][i];
        printf("Model Noleak (ANTS %i-%i): RR = (%.2e, %.2e); RL = (%.2e, %.2e); LR = (%.2e, %.2e); LL = (%.2e, %.2e)\n",A1[s][i],A2[s][i],RRAux.real(),RRAux.imag(),RLAux.real(),RLAux.imag(),LRAux.real(),LRAux.imag(),LLAux.real(),LLAux.imag());
        RRAux = RR/EMA[s][i];
        LLAux = LL*EMA[s][i];
        RLAux = RL/EPA[s][i];
        LRAux = LR*EPA[s][i];
        printf("Model Leak (ANTS %i-%i): RR = (%.2e, %.2e); RL = (%.2e, %.2e); LR = (%.2e, %.2e); LL = (%.2e, %.2e)\n",A1[s][i],A2[s][i],RRAux.real(),RRAux.imag(),RLAux.real(),RLAux.imag(),LRAux.real(),LRAux.imag(),LLAux.real(),LLAux.imag());

        printedGood += 1;
      };



// Residuals for each correlation product:
      resid[0] =  (DATA[s][iaux2  ] - RR);
      resid[1] =  (DATA[s][iaux2+1] - RL); 
      resid[2] =  (DATA[s][iaux2+2] - LR);
      resid[3] =  (DATA[s][iaux2+3] - LL);




// Vector of derivative*residuals (first, for source components):
      for(k=0; k<NSou[s]; k++){
        iaux4 = PSou[k+Ns];
        if(DEBUG && i == 5 && j == 0){printf("Sou pid (%i): %i  ",s,iaux4);};
        if(doDeriv && iaux4>=0){
          res = 0.0;
          for(l=0;l<4;l++){
            res += AllDer[iaux4][l].real()*resid[l].real()*F[l]*currWgt[l];
            res += AllDer[iaux4][l].imag()*resid[l].imag()*F[l]*currWgt[l];
          }; 
          DerVec[iaux4] += res; res = 0.0;
          for(l=0;l<4;l++){
            res += AllDer[iaux4+1][l].real()*resid[l].real()*F[l]*currWgt[l];
            res += AllDer[iaux4+1][l].imag()*resid[l].imag()*F[l]*currWgt[l];
          }; 
          DerVec[iaux4+1] += res; res = 0.0;
          if (doFaraday){
          for(l=0;l<4;l++){
            res += AllDer[iaux4+2][l].real()*resid[l].real()*F[l]*currWgt[l];
            res += AllDer[iaux4+2][l].imag()*resid[l].imag()*F[l]*currWgt[l];
          }; 
          DerVec[iaux4+2] += res; res = 0.0;
          for(l=0;l<4;l++){
            res += AllDer[iaux4+3][l].real()*resid[l].real()*F[l]*currWgt[l];
            res += AllDer[iaux4+3][l].imag()*resid[l].imag()*F[l]*currWgt[l];
          }; 
          DerVec[iaux4+3] += res; res = 0.0;
          };
        };
      };






// Derivatives w.r.t. Dterms:
      iaux4 = PAntR[A1[s][i]];
   //   currPar = 2*NtotSou;


      if(doDeriv && iaux4>=0){

        if(!FitPerIF || SpwId[s][i]==0){



        VisPar[currPar] = iaux4;
        VisPar[currPar+1] = iaux4+1;
        currPar += 2;
        if(DEBUG && i == 5 && j==0){printf("DR1 pid: %i/%i  ",iaux4,currPar);};
        
        AllDer[iaux4][1] = LLc + ord2*LRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j]);
        AllDer[iaux4+1][1] = Im*AllDer[iaux4][1];

        AllDer[iaux4][0] = LRc + ord2*LLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j]);
        AllDer[iaux4+1][0] = Im*AllDer[iaux4][0];

        
        res  = AllDer[iaux4][0].real()*resid[0].real()*F[0]*currWgt[0];
        res += AllDer[iaux4][0].imag()*resid[0].imag()*F[0]*currWgt[0];
        res += AllDer[iaux4][1].real()*resid[1].real()*F[1]*currWgt[1];
        res += AllDer[iaux4][1].imag()*resid[1].imag()*F[1]*currWgt[1];
        DerVec[iaux4] += res;

        res  = AllDer[iaux4+1][0].real()*resid[0].real()*F[0]*currWgt[0];
        res += AllDer[iaux4+1][0].imag()*resid[0].imag()*F[0]*currWgt[0];
        res += AllDer[iaux4+1][1].real()*resid[1].real()*F[1]*currWgt[1];
        res += AllDer[iaux4+1][1].imag()*resid[1].imag()*F[1]*currWgt[1];
        DerVec[iaux4+1] += res;

        };



        for(term=0;term<Nterms;term++){

        if(!FitPerIF || SpwId[s][i]==term+1){

          res = FitPerIF?1.0:NuPow[term][SpwId[s][i] + j*Nspw];
          iaux5 = iaux4 + 2 + 2*term;
          VisPar[currPar] = iaux5;
          VisPar[currPar+1] = iaux5+1;
          currPar += 2;
        //  if(DEBUG){printf("DR1 NuPow %i pid: %i/%i  ",term,iaux4,currPar);};
 
      //  printf("FitPerIF. ANT %i , %i %i\n",A2[s][i],currPar,iaux4);
 
  
      //    res = NuPow[term][SpwId[s][i] + j*Nspw];

          AllDer[iaux5][1] = (LLc + ord2*LRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j]))*res;
          AllDer[iaux5+1][1] = Im*AllDer[iaux5][1];

          AllDer[iaux5][0] = (LRc + ord2*LLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j]))*res;
          AllDer[iaux5+1][0] = Im*AllDer[iaux5][0];

        
          res  = AllDer[iaux5][0].real()*resid[0].real()*F[0]*currWgt[0];
          res += AllDer[iaux5][0].imag()*resid[0].imag()*F[0]*currWgt[0];
          res += AllDer[iaux5][1].real()*resid[1].real()*F[1]*currWgt[1];
          res += AllDer[iaux5][1].imag()*resid[1].imag()*F[1]*currWgt[1];
          DerVec[iaux5] += res;

          res  = AllDer[iaux5+1][0].real()*resid[0].real()*F[0]*currWgt[0];
          res += AllDer[iaux5+1][0].imag()*resid[0].imag()*F[0]*currWgt[0];
          res += AllDer[iaux5+1][1].real()*resid[1].real()*F[1]*currWgt[1];
          res += AllDer[iaux5+1][1].imag()*resid[1].imag()*F[1]*currWgt[1];
          DerVec[iaux5+1] += res;


          };



        };

      };

      iaux4 = PAntL[A1[s][i]];




      if(doDeriv && iaux4>=0){

        if(!FitPerIF || SpwId[s][i]==0){


        VisPar[currPar] = iaux4;
        VisPar[currPar+1] = iaux4+1;
        currPar += 2;
        if(DEBUG && i==5 && j==0){printf("DL1 pid: %i/%i  ",iaux4,currPar);};
        
        AllDer[iaux4][2] = RRc + ord2*RLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j]);
        AllDer[iaux4+1][2] = Im*AllDer[iaux4][2];

        AllDer[iaux4][3] = RLc + ord2*RRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j]);
        AllDer[iaux4+1][3] = Im*AllDer[iaux4][3];

        res  = AllDer[iaux4][2].real()*resid[2].real()*F[2]*currWgt[2];
        res += AllDer[iaux4][2].imag()*resid[2].imag()*F[2]*currWgt[2];
        res += AllDer[iaux4][3].real()*resid[3].real()*F[3]*currWgt[3];
        res += AllDer[iaux4][3].imag()*resid[3].imag()*F[3]*currWgt[3];
        DerVec[iaux4] += res;

        res  = AllDer[iaux4+1][2].real()*resid[2].real()*F[2]*currWgt[2];
        res += AllDer[iaux4+1][2].imag()*resid[2].imag()*F[2]*currWgt[2];
        res += AllDer[iaux4+1][3].real()*resid[3].real()*F[3]*currWgt[3];
        res += AllDer[iaux4+1][3].imag()*resid[3].imag()*F[3]*currWgt[3];
        DerVec[iaux4+1] += res;

        };

        for(term=0;term<Nterms;term++){

        if(!FitPerIF || SpwId[s][i]==term+1){

          res = FitPerIF?1.0:NuPow[term][SpwId[s][i] + j*Nspw];
          iaux5 = iaux4 + 2 + 2*term;
          VisPar[currPar] = iaux5;
          VisPar[currPar+1] = iaux5+1;
          currPar += 2;
        //  if(DEBUG){printf("DL1 NuPow %i pid: %i/%i  ",term,iaux4,currPar);};
       
     //     res = NuPow[term][SpwId[s][i] + j*Nspw];
          AllDer[iaux5][2] = (RRc + ord2*RLc*std::conj(DRp[A2[s][i]][SpwId[s][i]][j]))*res;
          AllDer[iaux5+1][2] = Im*AllDer[iaux5][2];

          AllDer[iaux5][3] = (RLc + ord2*RRc*std::conj(DLp[A2[s][i]][SpwId[s][i]][j]))*res;
          AllDer[iaux5+1][3] = Im*AllDer[iaux5][3];

        
          res  = AllDer[iaux5][2].real()*resid[2].real()*F[2]*currWgt[2];
          res += AllDer[iaux5][2].imag()*resid[2].imag()*F[2]*currWgt[2];
          res += AllDer[iaux5][3].real()*resid[3].real()*F[3]*currWgt[3];
          res += AllDer[iaux5][3].imag()*resid[3].imag()*F[3]*currWgt[3];
          DerVec[iaux5] += res;

          res  = AllDer[iaux5+1][2].real()*resid[2].real()*F[2]*currWgt[2];
          res += AllDer[iaux5+1][2].imag()*resid[2].imag()*F[2]*currWgt[2];
          res += AllDer[iaux5+1][3].real()*resid[3].real()*F[3]*currWgt[3];
          res += AllDer[iaux5+1][3].imag()*resid[3].imag()*F[3]*currWgt[3];
          DerVec[iaux5+1] += res;

          };

        };



      };





      iaux4 = PAntR[A2[s][i]];



      if(doDeriv && iaux4>=0){


        if(!FitPerIF || SpwId[s][i]==0){


        VisPar[currPar] = iaux4;
        VisPar[currPar+1] = iaux4+1;
        currPar += 2;
        if(DEBUG && i==5 && j==0){printf("DR2 pid: %i/%i  ",iaux4,currPar);};
      
        AllDer[iaux4][2] = LLc + ord2*RLc*DLp[A1[s][i]][SpwId[s][i]][j];
        AllDer[iaux4+1][2] = -Im*AllDer[iaux4][2];

        
        AllDer[iaux4][0] = RLc + ord2*LLc*DRp[A1[s][i]][SpwId[s][i]][j];
        AllDer[iaux4+1][0] = -Im*AllDer[iaux4][0];

        res  = AllDer[iaux4][0].real()*resid[0].real()*F[0]*currWgt[0];
        res += AllDer[iaux4][0].imag()*resid[0].imag()*F[0]*currWgt[0];
        res += AllDer[iaux4][2].real()*resid[2].real()*F[2]*currWgt[2];
        res += AllDer[iaux4][2].imag()*resid[2].imag()*F[2]*currWgt[2];
        DerVec[iaux4] += res;

        res  = AllDer[iaux4+1][0].real()*resid[0].real()*F[0]*currWgt[0];
        res += AllDer[iaux4+1][0].imag()*resid[0].imag()*F[0]*currWgt[0];
        res += AllDer[iaux4+1][2].real()*resid[2].real()*F[2]*currWgt[2];
        res += AllDer[iaux4+1][2].imag()*resid[2].imag()*F[2]*currWgt[2];
        DerVec[iaux4+1] += res;


        };

        for(term=0;term<Nterms;term++){

        if(!FitPerIF || SpwId[s][i]==term+1){

          res = FitPerIF?1.0:NuPow[term][SpwId[s][i] + j*Nspw];
          iaux5 = iaux4 + 2 + 2*term;

          VisPar[currPar] = iaux5;
          VisPar[currPar+1] = iaux5+1;
          currPar += 2;
       //   if(DEBUG){printf("DR2 NuPow %i pid: %i/%i  ",term,iaux4,currPar);};
       
      //    res = NuPow[term][SpwId[s][i] + j*Nspw];
          AllDer[iaux5][2] = (LLc + ord2*RLc*DLp[A1[s][i]][SpwId[s][i]][j])*res;
          AllDer[iaux5+1][2] = -Im*AllDer[iaux5][2];

          AllDer[iaux5][0] = (RLc + ord2*LLc*DLp[A1[s][i]][SpwId[s][i]][j])*res;
          AllDer[iaux5+1][0] = -Im*AllDer[iaux5][0];

        
          res  = AllDer[iaux5][0].real()*resid[0].real()*F[0]*currWgt[0];
          res += AllDer[iaux5][0].imag()*resid[0].imag()*F[0]*currWgt[0];
          res += AllDer[iaux5][2].real()*resid[2].real()*F[2]*currWgt[2];
          res += AllDer[iaux5][2].imag()*resid[2].imag()*F[2]*currWgt[2];
          DerVec[iaux5] += res;

          res  = AllDer[iaux5+1][0].real()*resid[0].real()*F[0]*currWgt[0];
          res += AllDer[iaux5+1][0].imag()*resid[0].imag()*F[0]*currWgt[0];
          res += AllDer[iaux5+1][2].real()*resid[2].real()*F[2]*currWgt[2];
          res += AllDer[iaux5+1][2].imag()*resid[2].imag()*F[2]*currWgt[2];
          DerVec[iaux5+1] += res;

        };

        };


      };




      iaux4 = PAntL[A2[s][i]];



      if(doDeriv && iaux4>=0){

        if(!FitPerIF || SpwId[s][i]==0){


        VisPar[currPar] = iaux4;
        VisPar[currPar+1] = iaux4+1;
        currPar += 2;
        if(DEBUG && i==5 && j==0){printf("DL2 pid: %i/%i  \n",iaux4,currPar);};

        AllDer[iaux4][1] = RRc + ord2*LRc*DRp[A1[s][i]][SpwId[s][i]][j];
        AllDer[iaux4+1][1] = -Im*AllDer[iaux4][1];

        AllDer[iaux4][3] = LRc + ord2*RRc*DLp[A1[s][i]][SpwId[s][i]][j];
        AllDer[iaux4+1][3] = -Im*AllDer[iaux4][3];

        res  = AllDer[iaux4][1].real()*resid[1].real()*F[1]*currWgt[1];
        res += AllDer[iaux4][1].imag()*resid[1].imag()*F[1]*currWgt[1];
        res += AllDer[iaux4][3].real()*resid[3].real()*F[3]*currWgt[3];
        res += AllDer[iaux4][3].imag()*resid[3].imag()*F[3]*currWgt[3];
        DerVec[iaux4] += res;

        res  = AllDer[iaux4+1][1].real()*resid[1].real()*F[1]*currWgt[1];
        res += AllDer[iaux4+1][1].imag()*resid[1].imag()*F[1]*currWgt[1];
        res += AllDer[iaux4+1][3].real()*resid[3].real()*F[3]*currWgt[3];
        res += AllDer[iaux4+1][3].imag()*resid[3].imag()*F[3]*currWgt[3];
        DerVec[iaux4+1] += res;


        };

        for(term=0;term<Nterms;term++){

        if(!FitPerIF || SpwId[s][i]==term+1){

          res = FitPerIF?1.0:NuPow[term][SpwId[s][i] + j*Nspw];
          iaux5 = iaux4 + 2 + 2*term;

          VisPar[currPar] = iaux5;
          VisPar[currPar+1] = iaux5+1;
          currPar += 2;
       //   if(DEBUG){printf("DL2 NuPow %i pid: %i/%i  ",term,iaux4,currPar);};
       
    //      res = NuPow[term][SpwId[s][i] + j*Nspw];
          AllDer[iaux5][1] = (RRc + ord2*LRc*DLp[A1[s][i]][SpwId[s][i]][j])*res;
          AllDer[iaux5+1][1] = -Im*AllDer[iaux5][1];

          AllDer[iaux5][3] = (LRc + ord2*RRc*DLp[A1[s][i]][SpwId[s][i]][j])*res;
          AllDer[iaux5+1][3] = -Im*AllDer[iaux5][3];

        
          res  = AllDer[iaux5][1].real()*resid[1].real()*F[1]*currWgt[1];
          res += AllDer[iaux5][1].imag()*resid[1].imag()*F[1]*currWgt[1];
          res += AllDer[iaux5][3].real()*resid[3].real()*F[3]*currWgt[3];
          res += AllDer[iaux5][3].imag()*resid[3].imag()*F[3]*currWgt[3];
          DerVec[iaux5] += res;

          res  = AllDer[iaux5+1][1].real()*resid[1].real()*F[1]*currWgt[1];
          res += AllDer[iaux5+1][1].imag()*resid[1].imag()*F[1]*currWgt[1];
          res += AllDer[iaux5+1][3].real()*resid[3].real()*F[3]*currWgt[3];
          res += AllDer[iaux5+1][3].imag()*resid[3].imag()*F[3]*currWgt[3];
          DerVec[iaux5+1] += res;

         };

        };


      };


      // Add up to the Chi Square:
      for(iaux4=0;iaux4<4;iaux4++){
        ChiSq += resid[iaux4].real()*resid[iaux4].real()*currWgt[iaux4]*F[iaux4];
        ChiSq += resid[iaux4].imag()*resid[iaux4].imag()*currWgt[iaux4]*F[iaux4];
      };

      
// Add up the product of derivatives to the Hessian matrix:
    if(DEBUG && i==5 && j==0){
    for (k=0;k<4;k++){  printf(" POL %i:\n",k);
      for (l=0;l<NPar;l++){ printf(" (% .3e, % .3ej) ",AllDer[l][k].real());  };
    };
   };

    if(doDeriv){
     for(k=0;k<currPar;k++){
       for(l=0;l<=k;l++){
         for(iaux4=0;iaux4<4;iaux4++){
           res  = AllDer[VisPar[k]][iaux4].real()*AllDer[VisPar[l]][iaux4].real()*F[iaux4]*currWgt[iaux4];
           res += AllDer[VisPar[k]][iaux4].imag()*AllDer[VisPar[l]][iaux4].imag()*F[iaux4]*currWgt[iaux4];
           Hessian[VisPar[k]*NPar + VisPar[l]] += res;

   //        if(DEBUG && VisPar[k]>39 && VisPar[l]>39 ){
   //   printf(" POL %i; DER %i / %i: (% .3e, % .3ej) | DER %i / %i: (% .3e, % .3ej) | H: % .3e \n",iaux4,k,VisPar[k],AllDer[VisPar[k]][iaux4].real(),AllDer[VisPar[k]][iaux4].imag(),l,VisPar[l],AllDer[VisPar[l]][iaux4].real(),AllDer[VisPar[l]][iaux4].imag(),Hessian[VisPar[k]*NPar + VisPar[l]]);
    //       };

         };
         if(k!=l){Hessian[VisPar[l]*NPar + VisPar[k]] = Hessian[VisPar[k]*NPar + VisPar[l]];};
       };
     };
    }; 

    if (doResid){ // Residuals for each correlation product:
      DATA[s][iaux2  ] = resid[0];
      DATA[s][iaux2+1] = resid[1]; 
      DATA[s][iaux2+2] = resid[2];
      DATA[s][iaux2+3] = resid[3];
    };

     };  // Comes from if(Wgt...)

    }; // Comes from loop over channels.
  };  // Comes from loop over visibilities

  }; // Comes from loop over calibrator sources.

// Reduced Chi Square:
  ChiSq /= (double)(Ndata);

// Ratio of Stokes I between model and data: 
//  for(s=0; s<NCalSour; s++){  
 //   IRatio[s] /= Iwt[s];  
// Update ratio (only if it is worth, i.e. changes above 0.1%):  
//    if (std::abs(IRatio[s] - 1.0) > 1.e-3){TotFlux[s] *= IRatio[s];}; 
//  };
  

  if (DEBUG){
    for (i=0;i<NPar;i++){
      for (j=0; j<NPar;j++){
        printf(" % .4e",Hessian[j*NPar+i]);
      };
      printf("\n");
    };
  };


// Return the reduced ChiSquare (and the extreme I ratios):
  Err = Py_BuildValue("[d,d]",ChiSq,TotFlux);

  NITER += 1;
  printf("\r ITER %i. ChiSq %.3e ; Model Factors: ",NITER,ChiSq);
  for (s=0; s<NCalSour;s++){printf(" %i -> %.3e ",s,IRatio[s]);};
  fflush(stdout);
  return Err;

};

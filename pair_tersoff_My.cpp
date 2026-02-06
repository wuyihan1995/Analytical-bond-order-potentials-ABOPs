/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL)


** 2022 - Feb:
   Modified by Yihan Wu, wyh6305076@stu.edu.cn.
   Adding equilibrium distance r_0 to the attraction and repulsion functions.
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_tersoff_My.h"   //  修改文件名
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

//  定义常量
#define MAXLINE 1024
#define DELTA 4



/* -------------------------------  构造函数  --------------------------------------- */
PairTersoffMy::PairTersoffMy(LAMMPS *lmp) : Pair(lmp)
{
  //  初始化参数
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;


  elements = NULL;             //  names of unique elements
  nelements = 0;               //  # of unique elements

  params = NULL;               //  parameter set for an I-J-K interaction
  nparams = 0;                 //  # of stored parameter *sets*，即当前对应的参数集下标
  maxparam = 0;                //  max # of parameter sets，即总的参数集个数

  map = NULL;                  //  mapping from atom types to elements
  elem2param = NULL;           //  mapping from element triplets to parameters

  maxshort = 10;               //  size of short neighbor list array
  neighshort = NULL;           //  short neighbor list array
}

/* -----------------------------  析构函数  ------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */
PairTersoffMy::~PairTersoffMy()
{
  if (copymode) return;

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  memory->destroy(params);
  memory->destroy(elem2param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
    delete [] map;
  }
}


















/*  ##################################################################################################
    ##################################################################################################
    ##      进行正式计算前对PairStyle的初始化设置以及*势函数参数的引入*
    ##################################################################################################
    ##################################################################################################*/

/*  ----------------------------------------------------------------------
    申请内存空间
   ------------------------------------------------------------------------  */
void PairTersoffMy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}
/* ----------------------------------------------------------------------
       ##  global settings  ##
------------------------------------------------------------------------- */
void PairTersoffMy::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}
/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairTersoffMy::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Tersoff requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Tersoff requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}
/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairTersoffMy::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}






/* ------------------------------------------------------------------------------------------------
    set coeffs for one or more type pairs

    确定element数组中元素和atomtype对应关系


	修正了原程序的bug。即Potential file中的元素顺序必须与pair_coeff中的元素顺序相同的bug --> (该bug仅存在于需要将参数与fix_qeq参数关联的势函数中，本势函数中没有该bug)


	程序中各个数组的对应关系：

		1. map[i] = j   ==>   i为AtomType，下标从1开始
		2. elements[j] = '元素名称'
		3. params[k].ielement = j  ==>  k为元素在potential file中的出现顺序，**当Potential file中
		   的元素顺序与pair_coeff中的相同，且pair_coeff命令中没有重复元素时**，k=j。即为bug来源。
		4. elem2param[j] = k

------------------------------------------------------------------------------------------------- */
void PairTersoffMy::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();
  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");
  // insure I,J args are * *
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");


  /*  ======================================================
      针对pair_coeff命令
    ======================================================  */
  //  read args that map atom types to elements in potential file
  //  map[i] = which element the Ith atom type is, -1 if NULL
  //  nelements = # of unique elements
  //  elements = list of UNIQUE element names  (不重复！！)  the sequence is the order they appear in the pair_coeff command.

  //  如果尚未清空初始化清空数组
  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  //  申请一个空数组，长度与atomtype数目相同（实际可能不会全部用到，只是为了防止数组超界）
  elements = new char*[atom->ntypes];
  //  初始化令所有元素为空
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;
  //  获得pair_coeff命令行中非重复元素的个数nelement；令map[atomtype]=j，element[j]='独立元素名'，若j=-1则对应NULL
  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;    //  按pair_coeff中的出现顺序，即atomtype
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;       //  按pair_coeff中的出现顺序，即atomtype

    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }


  /*  ======================================================
      针对势函数文件
    ======================================================  */
  // read potential file and initialize potential parameters
  read_file(arg[2]);
  setup_params();

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  //  set setflag i,j for type pairs where both are mapped to elements
  //  当元素i, j均不为NULL时，setflag=1
  int count = 0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }
  //  若count=0，则所有元素设置均为NULL，报错
  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}





/* ----------------------------------------------------------------------
    读取potential file，确定前三个参数与element数组中元素的对应关系

    前三个参数由元素名转换为整数ielement, jelement, kelement。
    这三个整数与element数组中的元素编号对应。

    arg[0:3]=['Fe', 'Co', 'Ni'], element[0:3]=['Ni', 'Co', 'Fe']   -->   ielement=2, jelement=1, kelement=0
  --------------------------------------------------------------------- */
void PairTersoffMy::read_file(char *file)
{
  //  Initialization
  int params_per_line = 18;                            //  修改，总参数数目为17+1=18，新增r0
  char **words = new char*[params_per_line+1];
  memory->sfree(params);
  params = NULL;
  nparams = 0;      //  当前对应的参数集下标
  maxparam = 0;     //  总参数集个数

  // open file on proc 0
  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open Tersoff potential file %s",file);
      error->one(FLERR,str);
    }
  }


  //  read each line out of file, skipping blank lines or leading '#'
  //  store line of params if all 3 element tags are in element list
  //  读取势函数文件的每一行，跳过空行与注释
  //  仅当前三个参数(即三个元素名)均为element数组中的元素时，才储存该行数据
  int n,nwords,ielement,jelement,kelement;
  char line[MAXLINE],*ptr;
  int eof = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // concatenate additional lines until have params_per_line words
    while (nwords < params_per_line) {
      n = strlen(line);
      if (comm->me == 0) {
        ptr = fgets(&line[n],MAXLINE-n,fp);
        if (ptr == NULL) {
          eof = 1;
          fclose(fp);
        } else n = strlen(line) + 1;
      }
      MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof) break;
      MPI_Bcast(&n,1,MPI_INT,0,world);
      MPI_Bcast(line,n,MPI_CHAR,0,world);
      if ((ptr = strchr(line,'#'))) *ptr = '\0';
      nwords = atom->count_words(line);
    }

    if (nwords != params_per_line)
      error->all(FLERR,"Incorrect format in Tersoff potential file");

    // words = ptrs to all words in line
    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;






    //  ielement,jelement,kelement = 1st, 2nd and 3rd args
    //  if all 3 args are in element list, then parse this line
    //  else skip to next line
    //  读取前三个参数，即元素名，当三个元素都在element数组中中才储存该行
    for (ielement = 0; ielement < nelements; ielement++)
      if (strcmp(words[0],elements[ielement]) == 0) break;   //  ielement即为满足words[0]=elements[ielement]
    if (ielement == nelements) continue;                     //  当element数组中无法找到与第一个参数相同的元素时，直接跳过该行
    for (jelement = 0; jelement < nelements; jelement++)
      if (strcmp(words[1],elements[jelement]) == 0) break;   //  jelement即为满足words[1]=elements[jelement]
    if (jelement == nelements) continue;                     //  当element数组中无法找到与第二个参数相同的元素时，直接跳过该行
    for (kelement = 0; kelement < nelements; kelement++)
      if (strcmp(words[2],elements[kelement]) == 0) break;   //  kelement即为满足words[2]=elements[kelement]
    if (kelement == nelements) continue;                     //  当element数组中无法找到与第三个参数相同的元素时，直接跳过该行





    //  load up parameter settings and error check their values
    //  当前参数集下标等于总参数集个数时，表明数组超界了，提升总参数集数目+DELTA(默认DELTA=4)
    if (nparams == maxparam) {
      maxparam += DELTA;
      params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                          "pair:params");
    }
    params[nparams].ielement = ielement;
    params[nparams].jelement = jelement;
    params[nparams].kelement = kelement;
    params[nparams].powerm = atof(words[3]);
    params[nparams].gamma = atof(words[4]);
    params[nparams].lam3 = atof(words[5]);
    params[nparams].c = atof(words[6]);
    params[nparams].d = atof(words[7]);
    params[nparams].h = atof(words[8]);
    params[nparams].powern = atof(words[9]);
    params[nparams].beta = atof(words[10]);
    params[nparams].lam2 = atof(words[11]);
    params[nparams].bigb = atof(words[12]);
    params[nparams].bigr = atof(words[13]);
    params[nparams].bigd = atof(words[14]);
    params[nparams].lam1 = atof(words[15]);
    params[nparams].biga = atof(words[16]);
    //  #########################################################################
    params[nparams].r0 = atof(words[17]);      //  修改，新增参数r0
    //  #########################################################################
    //  currently only allow m exponent of 1 or 3
    //  mint为对m取整
    params[nparams].powermint = int(params[nparams].powerm);


    //  #######################################################################

	// #######################################################################
    //  parameter sanity check
    if (params[nparams].c < 0.0 ||
        params[nparams].d < 0.0 ||
        params[nparams].powern < 0.0 ||
        params[nparams].beta < 0.0 ||
        params[nparams].lam2 < 0.0 ||
        params[nparams].bigb < 0.0 ||
        params[nparams].bigr < 0.0 ||
        params[nparams].bigd < 0.0 ||
        params[nparams].bigd > params[nparams].bigr ||
        params[nparams].lam1 < 0.0 ||
        params[nparams].biga < 0.0 ||
        params[nparams].powerm - params[nparams].powermint != 0.0 ||
        (params[nparams].powermint != 3 &&
         params[nparams].powermint != 1) ||
        params[nparams].gamma < 0.0  ||
        params[nparams].r0 < 0.0)
      error->all(FLERR,"Illegal Tersoff parameter");

    nparams++;
  }

  delete [] words;
}





/* --------------------------------------------------------------------------------
    对读取的参数进行进一步处理，参数集内每个单体对应的哪个元素

    elem2param[j1][j2][j3]=k，j为element数组中的下标，k为potential_file中对应参数集的行号（或者储存在param中的编号，如果不是每行都储存的话）
   --------------------------------------------------------------------------------  */
void PairTersoffMy::setup_params()
{
  int i,j,k,m,n;

  // set elem2param for all element triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  //  初始化数据
  memory->destroy(elem2param);
  memory->create(elem2param,nelements,nelements,nelements,"pair:elem2param");

  //  确定对应关系即elem2param数组中的元素
  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem2param[i][j][k] = n;
      }


  //  compute parameter values derived from inputs
  //  计算一些中间量
  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].bigr + params[m].bigd;                            //  截断半径为rc=R+D
    params[m].cutsq = params[m].cut*params[m].cut;                              //  rc^2


    //  所需的中间量，在后面会用到
    params[m].c1 = pow(2.0*params[m].powern*1.0e-16,-1.0/params[m].powern);     //  (2n*10^-16)^(-1/n)
    params[m].c2 = pow(2.0*params[m].powern*1.0e-8,-1.0/params[m].powern);      //  (2n*10^8)^(-1/n)
    params[m].c3 = 1.0/params[m].c2;                                            //  (2n*10^-16)^(1/n)
    params[m].c4 = 1.0/params[m].c1;                                            //  (2n*10^-8)^(1/n)
  }

  //  set cutmax to max of all params
  //  cutmax为所有triplet对应的截断半径(R+D)的最大值
  cutmax = 0.0;
  for (m = 0; m < nparams; m++)
    if (params[m].cut > cutmax) cutmax = params[m].cut;
}



















/*  ##################################################################################################
    ##################################################################################################
    ##                                         进行正式计算
    ##################################################################################################
    ##################################################################################################*/

/*  ----------------------------------------------------------------------
    计算能量与力
------------------------------------------------------------------------  */
void PairTersoffMy::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum;
  int itype,jtype,ktype,iparam_ij,iparam_ijk;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fi[3],fj[3],fk[3];
  double zeta_ij,prefactor;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = vflag_atom = 0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  const double cutshortsq = cutmax*cutmax;

  inum = list->inum;                  //  ${inum}th atom in the atom list
  ilist = list->ilist;                //  ilist ==> local indices of I atoms
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;


  //  -------------------------------------------------------------------------------------------
  //  --------------------  loop over full neighbor list of my atoms  ---------------------------
  //  -------------------------------------------------------------------------------------------
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];             //  i原子对应的atomtype对应的元素在element数组中的下标
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;


    //  +++++++++++++++  TWO-BODY interactions, skip half of them  +++++++++++++++++
    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;
    //  loop over all neighbors j of atom i
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      // i、j原子之间距离
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;  // 距离平方
      // cutshortsq=cutmax*cutmax
      if (rsq < cutshortsq) {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }
      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < x[i][2]) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }
      jtype = map[type[j]];
      iparam_ij = elem2param[itype][jtype][jtype];
      if (rsq >= params[iparam_ij].cutsq) continue;
      //%%%%%%%%%  调用repulsive函数计算排斥力  %%%%%%%%%
      repulsive(&params[iparam_ij],rsq,fpair,eflag,evdwl);
      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      /*---------------------------------------------------------
      力的计算方法是能量对原子*位置x*的导数
      由于这里rij = xi - xj
      因此fxtmp是+，fj是-
      ----------------------------------------------------------*/
      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;
      f[j][0] -= delx*fpair;
      f[j][1] -= dely*fpair;
      f[j][2] -= delz*fpair;
      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delx,dely,delz);
    }


    //  ++++++++++++++++++++  THREE-BODY interactions  +++++++++++++++++++++
    //  skip immediately if I-J is not within cutoff
    double fjxtmp,fjytmp,fjztmp;
    //  对第二原子j循环
    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      iparam_ij = elem2param[itype][jtype][jtype];
      //  r_ij = xj - xi
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      //  r_ij^2
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      if (rsq1 >= params[iparam_ij].cutsq) continue;

      //  accumulate bond-order zeta for each i-j interaction via loop over k
      //  %%%%%%%%%%%%%求解bond-order项b_ij%%%%%%%%%%%%%
      fjxtmp = fjytmp = fjztmp = 0.0;
      zeta_ij = 0.0;                                         //  bij = (1 + beta^n * zeta_ij^n)^(-1/2n)
      //  对第三原子k循环
      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;                              //  若j和k是同一原子则跳过
        k = neighshort[kk];
        ktype = map[type[k]];
        iparam_ijk = elem2param[itype][jtype][ktype];
        //  r_ik = xk - xi
        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        //  r_ik^2
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        //  若大于截断半径，则跳过
        if (rsq2 >= params[iparam_ijk].cutsq) continue;
        //  计算zeta_ij
        zeta_ij += zeta(&params[iparam_ijk],rsq1,rsq2,delr1,delr2);
      }

      // pairwise force due to zeta_ij(对所有zeta_ijk求和之后的结果)
      force_zeta(&params[iparam_ij],rsq1,zeta_ij,fpair,prefactor,eflag,evdwl);
      fxtmp += delr1[0]*fpair;
      fytmp += delr1[1]*fpair;
      fztmp += delr1[2]*fpair;
      fjxtmp -= delr1[0]*fpair;
      fjytmp -= delr1[1]*fpair;
      fjztmp -= delr1[2]*fpair;
      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,-fpair,-delr1[0],-delr1[1],-delr1[2]);
      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      //  attractive term via loop over k
      //  求解Va(r_ij)
      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        k = neighshort[kk];
        ktype = map[type[k]];
        iparam_ijk = elem2param[itype][jtype][ktype];
        //  r_ik
        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        //  r_ik^2
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        //  若大于截断半径，则跳过
        if (rsq2 >= params[iparam_ijk].cutsq) continue;
        //%%%%%%%%%  调用attractive函数计算吸引力  %%%%%%%%%
        attractive(&params[iparam_ijk],prefactor,rsq1,rsq2,delr1,delr2,fi,fj,fk);
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        /*--------------------------------------------------
        计算三体项的力时即用能量分别对xi, xj, xk求导即为fi, fj, fk
        ---------------------------------------------------*/
        fxtmp += fi[0];
        fytmp += fi[1];
        fztmp += fi[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];
        if (vflag_atom) v_tally3(i,j,k,fj,fk,delr1,delr2);
      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }
  //  -------------------------------------------------------------------------------------------
  //  -------------------------------------------------------------------------------------------
  //  -------------------------------------------------------------------------------------------


  if (vflag_fdotr) virial_fdotr_compute();
}






/* ----------------------------------------------------------------------
    对势作用中的额排斥力项
  ----------------------------------------------------------------------  */
void PairTersoffMy::repulsive(Param *param, double rsq, double &fforce,
                            int eflag, double &eng)
{
  double r,tmp_fc,tmp_fc_d,tmp_exp;

  r = sqrt(rsq);                                                           //  ij原子间距
  tmp_fc = ters_fc(r,param);                                               //  截断函数fc
  tmp_fc_d = ters_fc_d(r,param);                                           //  fc的导数 -- 对r求导
  tmp_exp = exp( -param->lam1 * (r-param->r0) );                           //  修改，Vr(r)  -->  Vr(r-r0)
  fforce = -param->biga * tmp_exp * (tmp_fc_d - tmp_fc*param->lam1) / r;   //  力表达式,无需修改
  if (eflag) eng = tmp_fc * param->biga * tmp_exp;                         //  能量表达式：Vr=fc*Aexp(-lam1*(r-r0))
}
/* ----------------------------------------------------------------------
    截断函数 fc= 1/2 - 1/2*sin(pi*(r-R)/2D)
  ---------------------------------------------------------------------  */
double PairTersoffMy::ters_fc(double r, Param *param)
{
  double ters_R = param->bigr;
  double ters_D = param->bigd;
  //  modified: allowing zero length of the cutoff region
  if (r <= ters_R-ters_D) return 1.0;
  if (r > ters_R+ters_D) return 0.0;
  return 0.5*(1.0 - sin(MY_PI2*(r - ters_R)/ters_D));  //  MY_PI2 --> Pi/2
}
/* ----------------------------------------------------------------------
    截断函数fc对r求导
  ---------------------------------------------------------------------  */
double PairTersoffMy::ters_fc_d(double r, Param *param)
{
  double ters_R = param->bigr;
  double ters_D = param->bigd;
  //  modified: allowing zero length of the cutoff region
  if (r <= ters_R-ters_D) return 0.0;
  if (r > ters_R+ters_D) return 0.0;
  return -(MY_PI4/ters_D) * cos(MY_PI2*(r - ters_R)/ters_D);
}
/* ----------------------------------------------------------------------
    计算zeta_ijk，zeta_ij=sum(zeta_ijk)|k

    bij = (1 + beta^n * zeta_ij^n)^(-1/2n)
  ---------------------------------------------------------------------  */
double PairTersoffMy::zeta(Param *param, double rsqij, double rsqik,
                         double *delrij, double *delrik)
{
  double rij,rik,costheta,arg,ex_delr;
  rij = sqrt(rsqij);
  rik = sqrt(rsqik);

  //  ##############  注意此处cos_theta的计算方法  ############
  costheta = (delrij[0]*delrik[0] + delrij[1]*delrik[1] +
              delrij[2]*delrik[2]) / (rij*rik);
  //  ########################################################

  if (param->powermint == 3) arg = pow(param->lam3 * (rij-rik),3.0);
  else arg = param->lam3 * (rij-rik);
  //  这里是避免大数超界
  if (arg > 69.0776) ex_delr = 1.e30;
  else if (arg < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(arg);

  return ters_fc(rik,param) * ters_gijk(costheta,param) * ex_delr;
}
/* ----------------------------------------------------------------------
    计算zeta_ij对r_ij导数
  ---------------------------------------------------------------------  */
void PairTersoffMy::force_zeta(Param *param, double rsq, double zeta_ij,
                             double &fforce, double &prefactor,
                             int eflag, double &eng)
{
  double r,fa,fa_d,bij;
  //  原子间距
  r = sqrt(rsq);
  //  计算Va(r_ij) = -Bexp(-lam2*(r-r0))
  fa = ters_fa(r,param);
  //  计算Va对r_ij导数
  fa_d = ters_fa_d(r,param);
  //  计算bond-order项b_ij
  bij = ters_bij(zeta_ij,param);
  //
  fforce = 0.5*bij*fa_d / r;
  prefactor = -0.5*fa * ters_bij_d(zeta_ij,param);
  if (eflag) eng = 0.5*bij*fa;
}
/* ----------------------------------------------------------------------
    计算Va(r_ij) = -Bexp(-lam2*(r-r0))
  ---------------------------------------------------------------------  */
double PairTersoffMy::ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  //  修改，加入r-r0
  return -param->bigb * exp(-param->lam2 * (r-param->r0)) * ters_fc(r,param);
}
/* ----------------------------------------------------------------------
    计算Va对r_ij导数
  ---------------------------------------------------------------------  */
double PairTersoffMy::ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  //  修改加入r0
  return param->bigb * exp(-param->lam2 * (r-param->r0)) *
    (param->lam2 * ters_fc(r,param) - ters_fc_d(r,param));
}
/* ----------------------------------------------------------------------
    计算bond-order项b_ij
  ---------------------------------------------------------------------  */
double PairTersoffMy::ters_bij(double zeta, Param *param)
{
  double tmp = param->beta * zeta;
  /*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  修改：
  这里的判断return是当beta*zeta过大或过小时的近似操作，注释掉不需要
  if (tmp > param->c1) return 1.0/sqrt(tmp);
  if (tmp > param->c2)
    return (1.0 - pow(tmp,-param->powern) / (2.0*param->powern))/sqrt(tmp);
  if (tmp < param->c4) return 1.0;
  if (tmp < param->c3)
    return 1.0 - pow(tmp,param->powern)/(2.0*param->powern);
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
  return pow(1.0 + pow(tmp,param->powern), -1.0/(2.0*param->powern));
}
/* ----------------------------------------------------------------------
    计算bond-order项b_ij对zeta_ij求导
  ---------------------------------------------------------------------  */
double PairTersoffMy::ters_bij_d(double zeta, Param *param)
{
  double tmp = param->beta * zeta;
  /*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  修改：
  这里的判断return是当beta*zeta过大或过小时的近似操作，注释掉不需要
  if (tmp > param->c1) return param->beta * -0.5*pow(tmp,-1.5);
  if (tmp > param->c2)
    return param->beta * (-0.5*pow(tmp,-1.5) *
                          // error in negligible 2nd term fixed 9/30/2015
                          // (1.0 - 0.5*(1.0 +  1.0/(2.0*param->powern)) *
                          (1.0 - (1.0 +  1.0/(2.0*param->powern)) *
                           pow(tmp,-param->powern)));
  if (tmp < param->c4) return 0.0;
  if (tmp < param->c3)
    return -0.5*param->beta * pow(tmp,param->powern-1.0);
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
  double tmp_n = pow(tmp,param->powern);
  return -0.5 * pow(1.0+tmp_n, -1.0-(1.0/(2.0*param->powern)))*tmp_n / zeta;
}








/* ----------------------------------------------------------------------
   attractive term
   use param_ij cutoff for rij test
   use param_ijk cutoff for rik test
------------------------------------------------------------------------- */
void PairTersoffMy::attractive(Param *param, double prefactor,
                             double rsqij, double rsqik,
                             double *delrij, double *delrik,
                             double *fi, double *fj, double *fk)
{
  double rij_hat[3],rik_hat[3];
  double rij,rijinv,rik,rikinv;
  //  rij_hat为单位矢量
  rij = sqrt(rsqij);
  rijinv = 1.0/rij;
  vec3_scale(rijinv,delrij,rij_hat);
  //  rik_hat为单位矢量
  rik = sqrt(rsqik);
  rikinv = 1.0/rik;
  vec3_scale(rikinv,delrik,rik_hat);
  //  prefactor = -0.5*fa * ters_bij_d(zeta_ij,param);
  //
  ters_zetaterm_d(prefactor,rij_hat,rij,rik_hat,rik,fi,fj,fk,param);
}
/* ----------------------------------------------------------------------

------------------------------------------------------------------------- */
void PairTersoffMy::ters_zetaterm_d(double prefactor,
                                  double *rij_hat, double rij,
                                  double *rik_hat, double rik,
                                  double *dri, double *drj, double *drk,
                                  Param *param)
{
  double gijk,gijk_d,ex_delr,ex_delr_d,fc,dfc,cos_theta,tmp;
  double dcosdri[3],dcosdrj[3],dcosdrk[3];

  fc = ters_fc(rik,param);
  dfc = ters_fc_d(rik,param);
  if (param->powermint == 3) tmp = pow(param->lam3 * (rij-rik),3.0);
  else tmp = param->lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (param->powermint == 3)
    ex_delr_d = 3.0*pow(param->lam3,3.0) * pow(rij-rik,2.0)*ex_delr;
  else ex_delr_d = param->lam3 * ex_delr;

  cos_theta = vec3_dot(rij_hat,rik_hat);
  gijk = ters_gijk(cos_theta,param);
  gijk_d = ters_gijk_d(cos_theta,param);
  costheta_d(rij_hat,rij,rik_hat,rik,dcosdri,dcosdrj,dcosdrk);
  /*---------------------------------------------------------
  力的计算方法是能量对原子*位置x*的导数
  由于这里rij = xj - xi, rik = xk - xi
  导数的第一项只有i是负值
  ----------------------------------------------------------*/
  // compute the derivative wrt Ri(即xi)
  // dri = -dfc*gijk*ex_delr*rik_hat;
  // dri += fc*gijk_d*ex_delr*dcosdri;
  // dri += fc*gijk*ex_delr_d*(rik_hat - rij_hat);

  vec3_scale(-dfc*gijk*ex_delr,rik_hat,dri);
  vec3_scaleadd(fc*gijk_d*ex_delr,dcosdri,dri,dri);
  vec3_scaleadd(fc*gijk*ex_delr_d,rik_hat,dri,dri);
  vec3_scaleadd(-fc*gijk*ex_delr_d,rij_hat,dri,dri);
  vec3_scale(prefactor,dri,dri);

  // compute the derivative wrt Rj
  // drj = fc*gijk_d*ex_delr*dcosdrj;
  // drj += fc*gijk*ex_delr_d*rij_hat;

  vec3_scale(fc*gijk_d*ex_delr,dcosdrj,drj);
  vec3_scaleadd(fc*gijk*ex_delr_d,rij_hat,drj,drj);
  vec3_scale(prefactor,drj,drj);

  // compute the derivative wrt Rk
  // drk = dfc*gijk*ex_delr*rik_hat;
  // drk += fc*gijk_d*ex_delr*dcosdrk;
  // drk += -fc*gijk*ex_delr_d*rik_hat;

  vec3_scale(dfc*gijk*ex_delr,rik_hat,drk);
  vec3_scaleadd(fc*gijk_d*ex_delr,dcosdrk,drk,drk);
  vec3_scaleadd(-fc*gijk*ex_delr_d,rik_hat,drk,drk);
  vec3_scale(prefactor,drk,drk);
}







/* ---------------------------------------------------------------------- */

void PairTersoffMy::costheta_d(double *rij_hat, double rij,
                             double *rik_hat, double rik,
                             double *dri, double *drj, double *drk)
{
  // first element is devative wrt Ri, second wrt Rj, third wrt Rk

  double cos_theta = vec3_dot(rij_hat,rik_hat);

  vec3_scaleadd(-cos_theta,rij_hat,rik_hat,drj);
  vec3_scale(1.0/rij,drj,drj);
  vec3_scaleadd(-cos_theta,rik_hat,rij_hat,drk);
  vec3_scale(1.0/rik,drk,drk);
  vec3_add(drj,drk,dri);
  vec3_scale(-1.0,dri,dri);
}

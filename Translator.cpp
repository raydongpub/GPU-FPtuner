// ROSE translator example: identity translator.
//
// No AST manipulations, just a simple translation:
//
//    input_code > ROSE AST > output_code
// To support parse cudaDeviceSynchronize(), modify the pre-include lirary at ROSE_Install_Instance/include-staging/cuda_HEADERS



//#include <rose.h>
#include "Translator.h"
#define TOTAL_OPERATORS 4


bool header_check(SgLocatedNode* lnode) {
  string suffix = Rose::StringUtility::fileNameSuffix(lnode->getFilenameString()); 
  if (suffix=="h" || suffix=="hpp" || suffix=="hh")
    return true;
  return false;
}

//class fpInstrument : public AstSimpleProcessing {
class fpInstrument{
  public:
    void visit(SgSourceFile* sfile);
    enum fp_op_kind {
      fp_add,
      fp_mul,
      fp_div,
      fp_sqrt,
      fp_unknown
    };
    typedef struct {
      // NOTE: operator order
      // cnt[0]: add
      // cnt[1]: mul
      // cnt[2]: div
      // cnt[3]: sqrt
      int nbop_cnt[TOTAL_OPERATORS];
      int ad_cnt[TOTAL_OPERATORS];
      int op_order[TOTAL_OPERATORS];
      int nbop;
      int order;
      // note: max_ad could be got when calculate ad for each operation type
      int max_ad;
    } fp_info;
    typedef struct {
      int order;
      int max_ad;
      int nbop;
    } func_info;
    fp_op_kind op_kind=fp_unknown;
    std::map<int, fp_info> loop_info;
    std::map<SgName, func_info> func_map;
    std::map<SgName, std::map<int, fp_info>> func_loop;
    std::vector<string> var_device;
    std::vector<string> func_visit;
    std::vector<string> host_func;
    string ben_dir;
    std::map<string, std::vector<string>> fvar_decl;
    //for loop container
    Rose_STL_Container<SgNode*> for_visited;
    int total_opt = TOTAL_OPERATORS;
    int fp_add_count=0;
    int fp_mul_count=0;
    int fp_div_count=0;
    int fp_sqrt_count=0;
    int fp_unknown_count=0;
    int cur_loop;
    bool has_fp_decl;
    SgName cur_f_name;
    void set_dir(string ben_d) {
      ben_dir = ben_d;
    }
    void print_fp_operation_count() {
      cout << "\nThe floating-point operation types:"
           << "\nAddition:       " << fp_add_count
           << "\nMultiplication: " << fp_mul_count
           << "\ndivision        " << fp_div_count
           << "\nsquare_root:    " << fp_sqrt_count 
           << "\nnon-support op: " << fp_unknown_count<< "\n\n";
      cout << "\nThe loop info:\n";
      for (std::map<int, fp_info>::iterator i=loop_info.begin();
           i!=loop_info.end(); i++) {
        cout << "\nid: " << i->first << " order: " << i->second.order << "\n"
             << "Addition       : " << i->second.nbop_cnt[0] 
             << " " << i->second.ad_cnt[0]
             << "\nMultiplication: " << i->second.nbop_cnt[1] 
             << " " << i->second.ad_cnt[1]
             << "\nDivision      : " << i->second.nbop_cnt[2] 
             << " " << i->second.ad_cnt[2]
             << "\nSquare_root   : " << i->second.nbop_cnt[3] 
             << " " << i->second.ad_cnt[3]
             << "\nnumber of op  : " << i->second.nbop 
             << "\nad of loop    : " << i->second.max_ad
             << "\norder of op   : "; 
        for (int j=0; j<TOTAL_OPERATORS; j++)    
          cout << i->second.op_order[j] << " ";
        cout << "\n\n";

      }

      for (std::map<SgName, std::map<int, fp_info>>::iterator i=func_loop.begin();
           i!=func_loop.end(); i++) {
        cout << "\nfunction     : " << i->first.getString() << " , order: " << func_map[i->first].order << " ,max_ad: " 
             << func_map[i->first].max_ad << " ,nbop: " << func_map[i->first].nbop;
        for (std::map<int, fp_info>::iterator j=i->second.begin();
             j!=i->second.end(); j++) {
          cout << "\nid             : " << j->first << " order: " << j->second.order
               << "\nAddition       : " << j->second.nbop_cnt[0] 
               << " " << j->second.ad_cnt[0]
               << "\nMultiplication: " << j->second.nbop_cnt[1] 
               << " " << j->second.ad_cnt[1]
               << "\nDivision      : " << j->second.nbop_cnt[2] 
               << " " << j->second.ad_cnt[2]
               << "\nSquare_root   : " << j->second.nbop_cnt[3] 
               << " " << j->second.ad_cnt[3]
               << "\nnumber of op  : " << j->second.nbop 
               << "\nad of loop    : " << j->second.max_ad
               << "\norder of op   : "; 
          for (int k=0; k<TOTAL_OPERATORS; k++)    
            cout << j->second.op_order[k] << " ";
          cout << "\n\n";
        }
      }
    };
    int forstatement(SgForStatement *forstmt, SgFunctionDeclaration* func_decl, int level, int id);
    void expr_profile(SgStatement* stmt, SgNode* node, SgFunctionDeclaration* func_decl, int loop_id);
    SgNode* compare_profile(SgNode* node);
    bool headerfile_check(SgLocatedNode* lnode);
    int ad_count(SgNode* node, SgStatement* stmt, SgBinaryOp* op, SgBinaryOp* stmt_bop, SgFunctionDeclaration* func_decl, int ad, int id, int op_id);
    bool isSameOp(SgBinaryOp* lop, SgBinaryOp* rop);
    int ad_stmt_count(SgNode* node, SgBinaryOp* nop, int ad);
    int ad_cont_count(SgExpression* cur_lhs, SgBinaryOp* cur_op, SgStatement* stmt, int ad);
    void tuning_sort(SgSourceFile* sfile);
    std::vector<int> m_sort(std::vector<int> vec, int start, int size, int sort_type);
    bool sort_cond(int lindex, int rindex, int sort_type);
    //tuning visit and precision tuning
    void tuning_visit(SgSourceFile* sfile, int func_iter, int cur_iter, int tune_op);
    void tuning_exe(SgForStatement *forstmt, int op_id);
    int tune_op(SgNode* node, SgBinaryOp* nop);
    //dynamic profiling for loops
    void loopsize_profile(SgSourceFile* sfile);
    void nestedloop_size(SgForStatement *forstmt, int level, int id);
    // functions related profiling
    void function_visit(SgSourceFile* sfile);
    void traverseInput(SgSourceFile* sfile);
    void TuneHigh(SgVariableDeclaration* decl, SgNode* node, int func_type, int var_type);
    void recur_set(SgType* array_type, string type, SgScopeStatement* scope);

};

bool fpInstrument::sort_cond(int lindex, int rindex, int sort_type) {
  std::map<SgName, func_info>::iterator lit = func_map.begin(); 
  std::map<SgName, func_info>::iterator rit = func_map.begin(); 
  switch(sort_type) {
  case 0:
    if (loop_info[cur_loop].ad_cnt[lindex] < loop_info[cur_loop].ad_cnt[rindex])
      return true;
    if (loop_info[cur_loop].nbop_cnt[lindex] < loop_info[cur_loop].nbop_cnt[rindex])
      return true;
    if (lindex > rindex && 
        loop_info[cur_loop].ad_cnt[lindex] == loop_info[cur_loop].ad_cnt[rindex] &&
        loop_info[cur_loop].nbop_cnt[lindex] == loop_info[cur_loop].nbop_cnt[rindex])
      return true;
    break;
  case 1:
    if (loop_info[lindex].max_ad < loop_info[rindex].max_ad)
      return true;
    if (loop_info[lindex].nbop < loop_info[rindex].nbop)
      return true;
    if (lindex < rindex && 
        loop_info[lindex].max_ad == loop_info[rindex].max_ad && 
        loop_info[lindex].nbop == loop_info[rindex].nbop)
      return true;
    break;
  case 2:
    for (int i=0; i<lindex; i++)
      lit++;
    for (int i=0; i<rindex; i++)
      rit++;
    if (lit->second.max_ad < rit->second.max_ad)
      return true;
    if (lit->second.nbop < rit->second.nbop)
      return true;
    if (lindex < rindex && 
        lit->second.max_ad == rit->second.max_ad && 
        lit->second.nbop   == rit->second.nbop)
      return true;
    break;
  case 3:
    if (func_loop[cur_f_name][cur_loop].ad_cnt[lindex] < func_loop[cur_f_name][cur_loop].ad_cnt[rindex])
      return true;
    if (func_loop[cur_f_name][cur_loop].nbop_cnt[lindex] < func_loop[cur_f_name][cur_loop].nbop_cnt[rindex])
      return true;
    if (lindex > rindex && 
        func_loop[cur_f_name][cur_loop].ad_cnt[lindex] == func_loop[cur_f_name][cur_loop].ad_cnt[rindex] &&
        func_loop[cur_f_name][cur_loop].nbop_cnt[lindex] == func_loop[cur_f_name][cur_loop].nbop_cnt[rindex])
      return true;
    break;
  case 4:
    if (func_loop[cur_f_name][lindex].max_ad < func_loop[cur_f_name][rindex].max_ad)
      return true;
    if (func_loop[cur_f_name][lindex].nbop < func_loop[cur_f_name][rindex].nbop)
      return true;
    if (lindex < rindex && 
        func_loop[cur_f_name][lindex].max_ad == func_loop[cur_f_name][rindex].max_ad && 
        func_loop[cur_f_name][lindex].nbop == func_loop[cur_f_name][rindex].nbop)
      return true;
    break;
  }
  return false;
}

bool fpInstrument::isSameOp(SgBinaryOp* lop, SgBinaryOp* rop) {

  if (lop->variantT() == V_SgAddOp || lop->variantT() == V_SgSubtractOp || 
      lop->variantT() == V_SgPlusAssignOp || lop->variantT() == V_SgMinusAssignOp)  {
    if (rop->variantT() == V_SgAddOp || rop->variantT() == V_SgSubtractOp || 
        rop->variantT() == V_SgPlusAssignOp || rop->variantT() == V_SgMinusAssignOp)  {
      return true;
    }
  }
  if (lop->variantT() == V_SgMultiplyOp || lop->variantT() == V_SgMultAssignOp) {
    if (rop->variantT() == V_SgMultiplyOp || rop->variantT() == V_SgMultAssignOp) {
      return true;
    }
  }
  if (lop->variantT() == V_SgDivideOp || lop->variantT() == V_SgDivAssignOp) {
    if (rop->variantT() == V_SgDivideOp || rop->variantT() == V_SgDivAssignOp) {
      return true;
    }
  }

  return false;
}
std::vector<int> fpInstrument::m_sort(std::vector<int> vec, int start, int size, int sort_type) {
  int i;
  int left = start;
  int lsize = size/2;
  int right = left + lsize;
  int rsize = size-size/2;
  std::vector<int> ret;
  std::vector<int> lret;
  std::vector<int> rret;
  ret.reserve(size);
  if (size > 1) {
    lret = m_sort(vec, left, lsize, sort_type);
    rret = m_sort(vec, right, rsize, sort_type);
    // merge
    int lptr = 0;
    int rptr = 0;
    for (i=0; i<size; i++) {
      if (lptr >= lsize) {
#ifdef _DEBUG
        printf("\tright: %d\n", rret[rptr]);
#endif
        ret[i] = rret[rptr];
        rptr++;
        continue;
      }
      if (rptr >= rsize) {
#ifdef _DEBUG
        printf("\tleft: %d\n", lret[lptr]);
#endif
        ret[i] = lret[lptr];
        lptr++;
        continue;
      }
      if (sort_cond(lret[lptr], rret[rptr], sort_type)) {
#ifdef _DEBUG
        printf("\tleft: %d\n", lret[lptr]);
#endif
        ret[i] = lret[lptr];
        lptr++;
      } else {
#ifdef _DEBUG
        printf("\tright: %d\n", rret[rptr]);
#endif
        ret[i] = rret[rptr];
        rptr++;
      }
    }
  } else {
#ifdef _DEBUG
    printf("\tsingle: %d\n", vec[start]);
#endif
    ret.push_back(vec[start]);
  }

  return ret; 
}

void fpInstrument::tuning_sort(SgSourceFile* sfile) {
  int num_loop = loop_info.size();
  int i, j, k;
  std::vector<int> loop_order;
  std::vector<int> opt_order;
  std::vector<int> ret_op;
  std::vector<int> func_order;
  std::vector<int> func_sorted;
  for (i=0; i<TOTAL_OPERATORS; i++)
    opt_order.push_back(i);
  for (i=0; i<num_loop; i++) {
    loop_order.push_back(i);
    cur_loop = i;
#ifdef _DEBUG
    printf("\ncur_loop: %d\n", cur_loop);
#endif
    ret_op = m_sort(opt_order, 0, TOTAL_OPERATORS, 0);
    for (j=0; j<TOTAL_OPERATORS; j++) {
      loop_info[i].op_order[j] = ret_op[j];
      opt_order[j] = j;
    }
  }
  loop_order = m_sort(loop_order, 0, num_loop, 1);
  for (i=0; i<num_loop; i++)
    loop_info[loop_order[i]].order = i;
  
  for (i=0; i<func_map.size(); i++)
    func_order.push_back(i);
  // sort loops in function map
  func_order = m_sort(func_order, 0, func_map.size(), 2);
  int func_size = func_map.size();
  for (int i=0; i<func_size; i++) {
       func_sorted.push_back(func_order[i]);
  }
  cout << "\n";
  //NOTE: fill function order
  #if 1
  num_loop = 0;
  for (std::map<SgName, std::map<int, fp_info>>::iterator i=func_loop.begin();
       i!=func_loop.end(); i++) {
    num_loop += i->second.size();
  }
  int counter = 0;
  std::vector<int> f_loop;
  for (std::map<SgName, std::map<int, fp_info>>::iterator i=func_loop.begin();
       i!=func_loop.end(); i++) {
    cur_f_name = i->first;
    //set function order in function map
    std::vector<int>::iterator it = std::find(func_sorted.begin(), func_sorted.end(), counter);
    int index = std::distance(func_sorted.begin(), it);
    func_map[cur_f_name].order = index;
    // sort order for operation types in a loop
    for (j=0; j<i->second.size(); j++) {
      f_loop.push_back(j);
      cur_loop = j;
      ret_op = m_sort(opt_order, 0, TOTAL_OPERATORS, 3);
      for (k=0; k<TOTAL_OPERATORS; k++) {
        func_loop[cur_f_name][j].op_order[k] = ret_op[k];
        opt_order[k] = k;
      }
    }
    f_loop = m_sort(f_loop, 0, i->second.size(), 4);
    for (j=0; j<i->second.size(); j++)
      func_loop[cur_f_name][f_loop[j]].order = j;
    counter++;
  }
  #endif
  
  // attach tuning plan to each loop
  #if 1
  int lid = 1;
  bool v_func = false;
  SgName cur_func_name = NULL;
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(sfile, V_SgStatement);
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    // retrieve statement
    SgStatement* stmt = isSgStatement(*i);
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    
    SgFunctionDefinition* cur_func_def = isSgFunctionDefinition(*i); 
    if (cur_func_def != NULL) {
      //cout << "\t\tStatement: " << (*i)->unparseToString() << "\n";
      cur_func_name = cur_func_def->get_declaration()->get_name();
      if (cur_func_name != NULL) {
        if (func_map.find(cur_func_name) != func_map.end()) {
          v_func = true;
          SgStatement* next_stmt = getFirstStatement(getScope(*i), false);
#ifdef _DEBUG
          cout << "\tfirst statement in func: " << ((SgNode*) next_stmt)->unparseToString() << "\n";
#endif
          lid = 1;
          // work on the new function
#ifdef _DEBUG
          cout << "\tinsert plan for func: " << cur_func_name.getString() << "\n";
#endif
          string func_text = "function " + Rose::StringUtility::numberToString(func_map[cur_func_name].order) + " " + 
                             Rose::StringUtility::numberToString(func_loop[cur_func_name].size());
          SgPragmaDeclaration* fpragmaDecl = buildPragmaDeclaration(func_text, NULL);
          insertStatementBefore(next_stmt, fpragmaDecl, true);
        } else
        v_func = false;
      }
    }
    // if the function is not in the function list, skip it
    if (!v_func)
      continue;

    if (stmt->variantT() == V_SgForStatement) {
      string prof_text = Rose::StringUtility::numberToString(func_loop[cur_func_name][lid].order) + " ";
      for(int j=0; j<TOTAL_OPERATORS; j++)
        prof_text += Rose::StringUtility::numberToString(func_loop[cur_func_name][lid].op_order[j]) + " " +
                     Rose::StringUtility::numberToString(func_loop[cur_func_name][lid].nbop_cnt[func_loop[cur_func_name][lid].op_order[j]]) + " ";
      // allocate pragma statement
      SgPragmaDeclaration* pragmaDecl = buildPragmaDeclaration(prof_text, NULL);
      insertStatementBefore(stmt, pragmaDecl, false);
      //attachComment(isSgLocatedNode(*i), prof_text, PreprocessingInfo::before, PreprocessingInfo::CpreprocessorUnknownDeclaration);
      lid++;
    }
  }
  #endif
}

int fpInstrument::ad_stmt_count(SgNode* node, SgBinaryOp* nop, int ad) {

  int ad_ret = ad;
  int ad_max = 0;
  SgBinaryOp* bop = nop;
  Rose_STL_Container<SgNode*> nodeList = node->get_traversalSuccessorContainer();
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    SgBinaryOp* iop = isSgBinaryOp(*i);
    if (iop == NULL) {
#ifdef _DEBUG
      cout << "\t\t node: " << (*i)->unparseToString() << ", none op\n";
#endif
      continue;
    }
    if (bop->variantT() == iop->variantT() || isSameOp(bop, iop)) {
      ad++;
      ad_ret = ad_stmt_count(*i, nop, ad);
#ifdef _DEBUG
      cout << "\t\t node: " << (*i)->unparseToString() << ", same op\n";
#endif
    }
    if (bop->variantT() != iop->variantT() && !isSameOp(bop, iop)) {
#ifdef _DEBUG
    cout << "\t\t node: " << (*i)->unparseToString() << ", diff op\n";
#endif
    }
    if (ad_ret > ad_max)
      ad_max = ad_ret;
  }
  return ad_ret;

}

int fpInstrument::ad_cont_count(SgExpression* cur_lhs, SgBinaryOp* cur_op, SgStatement* stmt, int ad) {

  int ad_max = ad;
  int init_ad = ad;
  SgNode* cur_node = (SgNode*) (cur_lhs);
  // check all usage of current statement
  // and get the maximum ad across the statements
  SgStatement* next = getNextStatement(stmt);
#ifdef _DEBUG
              cout << "current node: " << cur_node->unparseToString() << "\n";
#endif

  while (next != NULL) {
    // check if the next statement is expression
#ifdef _DEBUG
              cout << "\tgo through node: " << ((SgNode*)next)->unparseToString() << "\n";
#endif
    if (next->variantT() == V_SgExprStatement) {
#ifdef _DEBUG
              cout << "\texpr node: " << ((SgNode*)next)->unparseToString() << "\n";
#endif
      // get the expression information of next statement
      SgExprStatement* expr_stmt = (SgExprStatement*) next; 
      SgExpression* expr = expr_stmt->get_expression();
      SgNode* expr_node = (SgNode*) (expr_stmt); 

      Rose_STL_Container<SgNode*> nodeList;
      // check if the next statement rhs_operand has
      // the cur_lhs_operand and operation type is matched or not
      if (expr_node->get_numberOfTraversalSuccessors() > 0) {
#ifdef _DEBUG
              cout << "\tenough node: " << ((SgNode*)next)->unparseToString() << "\n";
#endif
        // binarynode of next statement
        SgBinaryOp* expr_bop = (SgBinaryOp*)(expr);
        // check if the next statement has the valid rhg_operand
        if (expr_bop == NULL) {
          next = getNextStatement(next);
          continue;
        }
        // if the left operand is the same, stop tracking
        SgExpression* expr_left = expr_bop->get_lhs_operand();
        if (getSymbolsUsedInExpression(expr_left)[0] == getSymbolsUsedInExpression(cur_lhs)[0]) 
          break;
#ifdef _DEBUG
              cout << "\tleft node: " << getSymbolsUsedInExpression(expr_left)[0]->get_name().getString() << "\n";
#endif
        // if not, continue to track ad
        SgExpression* expr_next = expr_bop->get_rhs_operand();
        if (expr_next == NULL) {
          next = getNextStatement(next);
          continue;
        }
#ifdef _DEBUG
              cout << "\tbinary node: " << ((SgNode*) expr_next)->unparseToString() << "\n";
#endif
        // get all the children nodes from rhs_operand in rhs_operand
        nodeList = expr_next->get_traversalSuccessorContainer();
        for (Rose_STL_Container<SgNode*>::iterator i=nodeList.begin(); i!=nodeList.end(); i++) {
          // traverae and retrieve all the variables used in children node
          //NOTE current: delete specfic flag for array, check curren segment
          //if (isSgPntrArrRefExp(*i)) {
            if (isSgExpression(*i) == NULL)
              continue;
            if (getSymbolsUsedInExpression(isSgExpression(*i)).size() == 0)
              continue;
            if (getSymbolsUsedInExpression(isSgExpression(*i))[0] == getSymbolsUsedInExpression(cur_lhs)[0]) {
#ifdef _DEBUG
              cout << "\ttarget node: " << (*i)->unparseToString() << "\n";
#endif
              SgNode* prev_node = (*i)->get_parent();
              SgBinaryOp* prev_bop;
              bool cont_ad = false;
              while (prev_node->get_parent() != expr_node) {
                prev_bop = isSgBinaryOp(prev_node); 
                if (prev_bop == NULL) {
                  prev_node = prev_node->get_parent();
                  continue;
                }
                // check if next statement has continuous operation type
                if (cur_op == NULL) {
                    cont_ad = true;
#ifdef _DEBUG
                    cout << "\t parent not continue.\n";
#endif
                } else {
                  if (prev_bop->variantT() == cur_op->variantT() || isSameOp(prev_bop, cur_op)) {
#ifdef _DEBUG
                    cout<< "\tparent node: " << prev_node->unparseToString() << "\n";
#endif
                    // if there is continuous op, ad++
                    ad++;
                  } else {
                    // if there is no continuous op, break and not following next statement anymore
                    cont_ad = true;
#ifdef _DEBUG
                    cout << "\t parent not continue.\n";
#endif
                    break;
                  }
                }
                  
                prev_node = prev_node->get_parent();
              }
              if (!cont_ad) {
                  // if there is continuous op and no other
                  // and no other ops, continue search ad 
                  // accumulation
#ifdef _DEBUG
                  cout << "\t parent continue: " << prev_node->unparseToString() << "\n";
#endif
                  SgExpression* next_lhs = expr_bop->get_lhs_operand(); 
                  ad = ad_cont_count(next_lhs, cur_op, next, ad); 
              }
            }
          //}
        }
      }
    }
    if (ad > ad_max)
      ad_max = ad;
    ad = init_ad;
    // go to next statement
    next = getNextStatement(next);
  }
  return ad_max;
}

// op is the operation type of current statement,
// required by determined if the following usage of 
// lhs-operand has the same operation
int fpInstrument::ad_count(SgNode* node, SgStatement* stmt, SgBinaryOp* op, SgBinaryOp* stmt_bop, SgFunctionDeclaration* func_decl, int ad, int id, int op_id) {
  
  int ad_ret = ad;
  int ad_cont = 0;
  // check the ad of current statement
  ad_ret += ad_stmt_count(node, isSgBinaryOp(node), ad);
  // add nbop for current operation count during ad calculation
  loop_info[id].nbop_cnt[op_id] += ad_ret;
  func_loop[func_decl->get_name()][id].nbop_cnt[op_id] += ad_ret;
  // check the ad with following statements
  if (stmt->variantT() == V_SgReturnStmt)
    return ad_ret;
  SgExpression* cur_lhs = stmt_bop->get_lhs_operand();
  SgNode* cur_node = (SgNode*) (cur_lhs);
  ad_cont = ad_cont_count(cur_lhs, op, stmt, ad_cont);
  ad_ret += ad_cont;

  return ad_ret;
}

bool fpInstrument::headerfile_check(SgLocatedNode* lnode) {
  string suffix = Rose::StringUtility::fileNameSuffix(lnode->getFilenameString()); 
  if (suffix=="h" || suffix=="hpp" || suffix=="hh")
    return true;
  return false;
}

SgNode* fpInstrument::compare_profile(SgNode* node) {
  SgBinaryOp* bop;
  SgNode* rhs_node = NULL;
  // check floating-point type for the entire program 
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(node, V_SgBinaryOp);
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    bop = isSgBinaryOp(*i);
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    // check floating-point operation type
    switch (bop->variantT())
    {
      //case V_SgPlusAssignOp:
      //case V_SgMinusAssignOp:
      case V_SgEqualityOp:
      case V_SgNotEqualOp:
      case V_SgGreaterOrEqualOp:
      case V_SgGreaterThanOp:
      case V_SgLessOrEqualOp:
      case V_SgLessThanOp:
        // store operator type into loop_profiling
        rhs_node = (SgNode*) (bop->get_rhs_operand_i());
        break;
      default:
        //printf("Error: not supported operations for record\n");
        break;
    }
  }
  return rhs_node;
}

void fpInstrument::expr_profile(SgStatement* stmt, SgNode* node, SgFunctionDeclaration* func_decl, int loop_id){
  // check floating-point type of given expression statement
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(node, V_SgBinaryOp);
  SgNode* next;
  SgExprStatement* expr_stmt;
  SgNode* expr_node;
  SgBinaryOp* assign_op = NULL;
  SgBinaryOp* stmt_op;
  SgBinaryOp* prev_op = NULL;
  SgNode* prev_node = NULL;
  SgNode* func_expr;
  string func_support;
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    int ad = 0;
    SgBinaryOp* bop = isSgBinaryOp(*i);
    op_kind = fp_unknown;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    //NOTE: some floating-point opeartions are not recognized as floating type
#ifdef _DEBUG
    cout << "\t Op: " << (*i)->unparseToString() << "\n";
#endif
    // if not floating-point operations, skip it.
    if (!(bop->get_type()->isFloatType()))
      continue;
    // check floating-point operation type
    switch (bop->variantT())
    {
      case V_SgAssignOp:
#ifdef _DEBUG
        cout << "\n Assign Op: " << (*i)->unparseToString() << "\n";
#endif
        stmt_op = bop;
        prev_op = NULL;
        func_expr = (SgNode*) bop->get_rhs_operand();
        if (bop->get_rhs_operand()->get_numberOfTraversalSuccessors() > 1 && isSgFunctionCallExp(func_expr) == NULL) {
          next = *(i+2);
          assign_op = isSgBinaryOp(next);
        } else {
          //if (isSgFunctionCallExp(func_expr)->getAssociatedFunctionSymbol()->get_name().getString() == "sqrt") {
           #if 0
          if (isSgFunctionCallExp(func_expr)->class_name() == "sqrt") {
            cout << "\n Assign op with sqrt\n";
            func_support = isSgFunctionCallExp(func_expr)->class_name();
          }
          #endif
          assign_op = NULL;
        }
        break;
      case V_SgPlusAssignOp:
      case V_SgMinusAssignOp:
        //assignment operation settings
#ifdef _DEBUG
        cout << "\n Plus/Sub Assign Op: " << (*i)->unparseToString() << "\n";
#endif
        stmt_op = bop;
        prev_op = bop;
        assign_op = bop;
        // profile addition
        ad = ad_count(*i, stmt, assign_op, stmt_op, func_decl, ad, loop_id, 0);
        op_kind = fp_add;
        fp_add_count++;
        // store operator type into loop_profiling
        loop_info[loop_id].nbop_cnt[0]++;
        func_loop[func_decl->get_name()][loop_id].nbop_cnt[0]++;
        if (ad > func_loop[func_decl->get_name()][loop_id].ad_cnt[0]) {
          loop_info[loop_id].ad_cnt[0] = ad;
          func_loop[func_decl->get_name()][loop_id].ad_cnt[0] = ad;
        }
        if (ad > func_loop[func_decl->get_name()][loop_id].max_ad) {
          loop_info[loop_id].max_ad = ad;
          func_loop[func_decl->get_name()][loop_id].max_ad = ad;
        }
        break;
      case V_SgAddOp:
      case V_SgSubtractOp:
        if (prev_op != NULL) {
          if ((prev_op->variantT() == bop->variantT() || isSameOp(prev_op, bop)) && (*i)->get_parent() == prev_node)
            break;
        }
#ifdef _DEBUG
        cout << "\n The add following by: \n";
        cout << "\t add expr: " << (*i)->unparseToString() << "\n";
        if (assign_op != NULL) {
          if ((assign_op->variantT() == V_SgAddOp || assign_op->variantT() == V_SgSubtractOp))
            cout << "add assignment.\n";
        }
#endif
        ad = ad_count(*i, stmt, assign_op, stmt_op, func_decl, ad, loop_id, 0);
        op_kind = fp_add;
        fp_add_count++;
#ifdef _DEBUG
        cout << "ad: " << ad << "\n";
#endif
        // store operator type into loop_profiling
        loop_info[loop_id].nbop_cnt[0]++;
        func_loop[func_decl->get_name()][loop_id].nbop_cnt[0]++;
        if (ad > func_loop[func_decl->get_name()][loop_id].ad_cnt[0]) {
          loop_info[loop_id].ad_cnt[0] = ad;
          func_loop[func_decl->get_name()][loop_id].ad_cnt[0] = ad;
        }
        if (ad > func_loop[func_decl->get_name()][loop_id].max_ad) {
          loop_info[loop_id].max_ad = ad;
          func_loop[func_decl->get_name()][loop_id].max_ad = ad;
        }
        // avoid visit subtree of the same operation
        prev_op = bop;
        prev_node = *i;
        break;
      case V_SgMultAssignOp:
        //assignment operation settings
#ifdef _DEBUG
        cout << "\n Mul Assign Op: " << (*i)->unparseToString() << "\n";
#endif
        stmt_op = bop;
        prev_op = bop;
        assign_op = bop;
        //multiplication operation settings
        ad = ad_count(*i, stmt, assign_op, stmt_op, func_decl, ad, loop_id, 1);
        op_kind = fp_mul;
        fp_mul_count++;
#ifdef _DEBUG
        cout << "ad: " << ad << "\n";
#endif
        // store operator type into loop_profiling
        loop_info[loop_id].nbop_cnt[1]++;
        func_loop[func_decl->get_name()][loop_id].nbop_cnt[1]++;
        if (ad > func_loop[func_decl->get_name()][loop_id].ad_cnt[1]) {
          loop_info[loop_id].ad_cnt[1] = ad;
          func_loop[func_decl->get_name()][loop_id].ad_cnt[1] = ad;
        }
        if (ad > func_loop[func_decl->get_name()][loop_id].max_ad) {
          loop_info[loop_id].max_ad = ad;
          func_loop[func_decl->get_name()][loop_id].max_ad = ad;
        }
        break;
      case V_SgMultiplyOp:
        if (prev_op != NULL) {
          if ((prev_op->variantT() == bop->variantT() || isSameOp(prev_op, bop)) && (*i)->get_parent() == prev_node)
            break;
        }
#ifdef _DEBUG
        cout << "\n The mul following by: \n";
        if (assign_op != NULL) {
          if (assign_op->variantT() == V_SgMultiplyOp || assign_op->variantT() == V_SgMultAssignOp)
            cout << "mul assignment.\n";
        }
#endif
        ad = ad_count(*i, stmt, assign_op, stmt_op, func_decl, ad, loop_id, 1);
        op_kind = fp_mul;
        fp_mul_count++;
#ifdef _DEBUG
        cout << "ad: " << ad << "\n";
#endif
        // store operator type into loop_profiling
        loop_info[loop_id].nbop_cnt[1]++;
        func_loop[func_decl->get_name()][loop_id].nbop_cnt[1]++;
        if (ad > func_loop[func_decl->get_name()][loop_id].ad_cnt[1]) {
          loop_info[loop_id].ad_cnt[1] = ad;
          func_loop[func_decl->get_name()][loop_id].ad_cnt[1] = ad;
        }
        if (ad > func_loop[func_decl->get_name()][loop_id].max_ad) {
          loop_info[loop_id].max_ad = ad;
          func_loop[func_decl->get_name()][loop_id].max_ad = ad;
        }
        // avoid visit subtree of the same operation
        prev_op = bop;
        prev_node = *i;
        break;
      case V_SgDivAssignOp:
        //assignment operation settings
#ifdef _DEBUG
        cout << "\n Div Assign Op: " << (*i)->unparseToString() << "\n";
#endif
        stmt_op = bop;
        prev_op = bop;
        assign_op = bop;
        //division operation settings
        ad = ad_count(*i, stmt, assign_op, stmt_op, func_decl, ad, loop_id, 2);
        op_kind = fp_div;
        fp_div_count++;
#ifdef _DEBUG
        cout << "ad: " << ad << "\n";
#endif
        // store operator type into loop_profiling
        loop_info[loop_id].nbop_cnt[2]++;
        func_loop[func_decl->get_name()][loop_id].nbop_cnt[2]++;
        if (ad > func_loop[func_decl->get_name()][loop_id].ad_cnt[2]) {
          loop_info[loop_id].ad_cnt[2] = ad;
          func_loop[func_decl->get_name()][loop_id].ad_cnt[2] = ad;
        }
        if (ad > func_loop[func_decl->get_name()][loop_id].max_ad) {
          loop_info[loop_id].max_ad = ad;
          func_loop[func_decl->get_name()][loop_id].max_ad = ad;
        }
        break;
      case V_SgDivideOp:
        if (prev_op != NULL) {
          if (prev_op->variantT() == bop->variantT() || isSameOp(prev_op, bop))
            break;
        }
#ifdef _DEBUG
        cout << "\n The div following by: \n";
        if (assign_op != NULL) {
          if ((assign_op->variantT() == V_SgDivideOp || assign_op->variantT() == V_SgDivAssignOp) && (*i)->get_parent() == prev_node)
            cout << "div assignment.\n";
        }
#endif
        ad = ad_count(*i, stmt, assign_op, stmt_op, func_decl, ad, loop_id, 2);
        op_kind = fp_div;
        fp_div_count++;
#ifdef _DEBUG
        cout << "ad: " << ad << "\n";
#endif
        // store operator type into loop_profiling
        loop_info[loop_id].nbop_cnt[2]++;
        func_loop[func_decl->get_name()][loop_id].nbop_cnt[2]++;
        if (ad > func_loop[func_decl->get_name()][loop_id].ad_cnt[2]) {
          loop_info[loop_id].ad_cnt[2] = ad;
          func_loop[func_decl->get_name()][loop_id].ad_cnt[2] = ad;
        }
        if (ad > func_loop[func_decl->get_name()][loop_id].max_ad) {
          loop_info[loop_id].max_ad = ad;
          func_loop[func_decl->get_name()][loop_id].max_ad = ad;
        }
        // avoid visit subtree of the same operation
        prev_op = bop;
        prev_node = *i;
        break;
      default:
        op_kind = fp_unknown;
        fp_unknown_count++;
        //printf("Error: not supported operations for record\n");
        break;
    }
  }

}

int fpInstrument::forstatement(SgForStatement *forstmt, SgFunctionDeclaration* func_decl, int level, int id) {
  int num_stmt = 4;
  int nested_num_stmt = 0;
  int loop_id = 0;
  fp_info init_info = {{0, 0, 0, 0}, {0, 0, 0, 0}, {3, 2, 1, 0}, 0, 0, 0};
  // get loop body
  SgStatement* forbodystmt = forstmt->get_loop_body(); 
  SgType* type;
  SgType* def_type;
  SgTypedefType* tydef;
  SgVariableDeclaration* varsdeclaration;
  SgInitializedName* def_name;
  string cur_var_name;
  // check floating-point instruction or other loops inside loop body
  #if 0
  Rose_STL_Container<SgNode*> nodeList = forbodystmt->get_traversalSuccessorContainer();
  #else
  SgNode* bodynode = (SgNode*) forbodystmt;
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(bodynode, V_SgStatement);
  int num_children = forbodystmt->get_numberOfTraversalSuccessors();
  int count = 0;
  #endif
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++) {
    count++;
#ifdef _DEBUG
    cout << "\t check fornode: " << (*i)->unparseToString() << " , num: " << forbodystmt->get_numberOfTraversalSuccessors() << " , scope: " << getScope(*i)->class_name().c_str() <<"\n" ;
#endif
    SgStatement* stmt = isSgStatement(*i);
    SgExprStatement* expr_stmt = isSgExprStatement(*i);
    SgExpression* expr;
    if (stmt == NULL)
      continue;
    num_stmt++;
    switch (stmt->variantT())
    {
      case V_SgExprStatement:
#ifdef _DEBUG
        cout << "\texpr node, " << "level: " << level 
             << " stmt: " << (*i)->unparseToString() << "\n" ;
#endif
        expr_profile(stmt, *i, func_decl, id);
        break;
      case V_SgForStatement:
        // loop level increment for nested loop
        level++;
        // get loop_id for new identified loop, and add it into function map
        //loop_id = loop_info.size();
        loop_id = func_loop[func_decl->get_name()].size();
        loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
        func_loop[func_decl->get_name()][loop_id] = init_info;
        //check nested loop
        if (isSgForStatement(*i) != NULL) {
#ifdef _DEBUG
          cout << "\tfor node, " << "id, " << loop_id << " level: " 
               << level << " stmt: " << (*i)->unparseToString() << "\n" ;
#endif
          nested_num_stmt += forstatement(isSgForStatement(*i), func_decl, level, loop_id);
          i += nested_num_stmt;
        }
        num_stmt += nested_num_stmt;
        break;
      case V_SgVariableDeclaration:
        varsdeclaration = isSgVariableDeclaration(*i);
        type =  varsdeclaration->get_definition()->get_type();
        def_name = varsdeclaration->get_definition()->get_vardefn();
        cur_var_name = def_name->get_name().getString();
        if (type->hasExplicitType()) {
          def_type = type->dereference();
          tydef    = isSgTypedefType(def_type);
          if (def_name->get_type()->dereference()->isFloatType()) {
            has_fp_decl = true;
            fvar_decl[func_decl->get_name().getString()].push_back(cur_var_name); 
          } else if(tydef != NULL && tydef->get_name().getString() == "double3" ) {
            has_fp_decl = true;
            fvar_decl[func_decl->get_name().getString()].push_back(cur_var_name); 
          }
        } else if (type->isFloatType()) {
          has_fp_decl = true;
          fvar_decl[func_decl->get_name().getString()].push_back(cur_var_name); 
        } else if (isSgTypedefType(type) != NULL && isSgTypedefType(type)->get_name().getString() == "double3" ) {
          has_fp_decl = true;
          fvar_decl[func_decl->get_name().getString()].push_back(cur_var_name); 
        }
        break;
      default:
        if (isSgPrintStatement(*i) != NULL)
          cout << "\t\t Not recognized statement: " << (*i)->unparseToString() << "\n";
        //printf("Error: not supported operations for record\n");
        break;
    }
    if (count == num_children+1)
      break;
  }
  // delete extra stmt from
  num_stmt--;
  // calculate the total number of operations inside current loop
  int i;
  for (i=0; i<total_opt; i++) {
    loop_info[id].nbop += loop_info[id].nbop_cnt[i];
    func_loop[func_decl->get_name()][id].nbop += func_loop[func_decl->get_name()][id].nbop_cnt[i];
  }

  return num_stmt;
}

void fpInstrument::visit(SgSourceFile* sfile) {

  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(sfile, V_SgStatement);
  // build outer loop profling container and insert it to loop_info
  int loop_id = 0;
  int outloop_id = 0;
  assert(loop_id == outloop_id);
  fp_info init_info = {{0, 0, 0, 0}, {0, 0, 0, 0}, {3, 2, 1, 0}, 0, 0, 0};
  SgStatement* loop_cond;
  SgNode* loop_size_var;
  SgName cur_func_name = NULL;
  SgFunctionDeclaration* cur_func_decl;
  SgVariableDeclaration* varsdeclaration;
  SgExpression* Ret_expr;
  SgType* type;
  SgType* def_type;
  SgTypedefType* tydef;
  SgInitializedName* def_name;
  string cur_var_name;
  bool v_func = false;
  has_fp_decl = false;

  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    // retrieve statement
    SgStatement* stmt = isSgStatement(*i);
    // loop condition statement
    int loop_level = 0;
    int num_skip_stmt = 0;
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    // check function
    // check current function
    SgFunctionDefinition* cur_func_def = isSgFunctionDefinition(*i); 
    if (cur_func_def != NULL) {
      //cout << "\t\tStatement: " << (*i)->unparseToString() << "\n";
      if (func_map.find(cur_func_def->get_declaration()->get_name()) != func_map.end() && cur_func_name != cur_func_def->get_declaration()->get_name()) {
        v_func = true;
        // fill completed function profiling result
        if (cur_func_name != NULL)   {
#ifdef _DEBUG
          cout << "\tcurrent function: " << cur_func_name.getString() << ", next function: " << cur_func_def->get_declaration()->get_name().getString() << "\n";
#endif
          //update the function list of current function, variable_list and order
          //update function name
          if (fvar_decl.find(cur_func_name.getString()) != fvar_decl.end()) {
            string open_name = ben_dir+"compute_func.txt";
        		std::ofstream file(open_name, std::ofstream::out|std::ofstream::app);
        		file << cur_func_name.getString() << ":" << std::endl;
            for (std::vector<string>::iterator j=fvar_decl[cur_func_name.getString()].begin();
                   j!=fvar_decl[cur_func_name.getString()].end(); j++) {
              file << *j << std::endl;
            }
            file << std::endl;
            //add vardecl out of loop and inside loop

          } else {
            string open_name = ben_dir+"call_func.txt";
        		std::ofstream file(open_name, std::ofstream::out|std::ofstream::app);
        		file << cur_func_name.getString() << ":" << std::endl;
            file << std::endl;
            //add variable names that call this function
          }

          int f_loop_size = func_loop[cur_func_name].size();
          // update profile info for out of loop operations
          for (int j=0; j<total_opt; j++) {
            loop_info[0].nbop += loop_info[0].nbop_cnt[j];
            func_loop[cur_func_name][0].nbop += func_loop[cur_func_name][0].nbop_cnt[j];
          }
          for (int j=0; j<f_loop_size; j++) {
            // get max0mum ad for a completed function based on the loop profiling result
#ifdef _DEBUG
            cout << "\t\t max_ad: " << func_loop[cur_func_name][j].max_ad << ", map_ad: " << func_map[cur_func_name].max_ad << "\n";
#endif
            if (func_loop[cur_func_name][j].max_ad > func_map[cur_func_name].max_ad)
              func_map[cur_func_name].max_ad = func_loop[cur_func_name][j].max_ad;
            // get total nbop for a completed function
            func_map[cur_func_name].nbop += func_loop[cur_func_name][j].nbop;
#ifdef _DEBUG
            cout << "\t\t nbop: " << func_loop[cur_func_name][j].nbop << ", map_nbop: " << func_map[cur_func_name].nbop << "\n";
#endif

          }
        }
        // work on the new function
        has_fp_decl = false;
        cur_func_decl = cur_func_def->get_declaration();
        cur_func_name = cur_func_def->get_declaration()->get_name();
        cout << "\tfunc name: " << cur_func_name.getString() << "\n";
        // add loop info into func_loop map 
        loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
        func_loop[cur_func_name][loop_id] = init_info;
      } else
        v_func = false;
        has_fp_decl = false;
    }
    // if the function is not in the function list, skip it
    if (!v_func)
      continue;
    // check statement type to get loops or normal flops
    switch (stmt->variantT())
    {
    #ifdef _DEBUG
      // statements for declaration, definition and flops
      case V_SgExprStatement:
        expr_profile(stmt, *i, cur_func_decl, outloop_id);
#ifdef _DEBUG
        cout << "expr node, " << "level: " << loop_level 
             << " stmt: " << (*i)->unparseToString() << "\n" ;
#endif
        break;
      // statements for for loops
      case V_SgForStatement:
        // loop count increment and assign loop id to new identified loop
        loop_level++;
        // get loop size variable 
        loop_cond = getLoopCondition(isSgScopeStatement(*i));
        loop_size_var = compare_profile(loop_cond->get_traversalSuccessorByIndex(0));
        // get loop_id for new identified loop, add the loop into function map
        loop_id = func_loop[cur_func_name].size();
        loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
        func_loop[cur_func_name][loop_id] = init_info;

#ifdef _DEBUG
        cout << "for node, " << "id: " << loop_id <<", level: " 
             << loop_level << " size: " 
             << loop_size_var->unparseToString() << " stmt: " 
             << (*i)->unparseToString() << "\n" ;
#endif
        // profile loop information
        if (isSgForStatement(*i) != NULL) {
          num_skip_stmt += forstatement(isSgForStatement(*i), cur_func_decl, loop_level, loop_id);
#ifdef _DEBUG
          cout << "\t number of stmt: " << num_skip_stmt << "\n" ;
#endif
        }
        i += num_skip_stmt;
        break;
      case V_SgReturnStmt:
        cout << "Return stmt: " << (*i)->unparseToString()  << "\n";
        Ret_expr = isSgReturnStmt(*i)->get_expression();
        expr_profile(stmt, *i, cur_func_decl, outloop_id);
        break;
    #endif
      case V_SgVariableDeclaration:
        varsdeclaration = isSgVariableDeclaration(*i);
        type =  varsdeclaration->get_definition()->get_type();
        def_name = varsdeclaration->get_definition()->get_vardefn();
        cur_var_name = def_name->get_name().getString();
        if (type->hasExplicitType()) {
          def_type = type->dereference();
          tydef    = isSgTypedefType(def_type);
          if (def_name->get_type()->dereference()->isFloatType()) {
            cout << "\tvar decl: " << (*i)->unparseToString() << "\n";
            has_fp_decl = true;
            fvar_decl[cur_func_name.getString()].push_back(cur_var_name); 
          } else if(tydef != NULL && tydef->get_name().getString() == "double3" ) {
            cout << "\tvar decl: " << (*i)->unparseToString() << "\n";
            has_fp_decl = true;
            fvar_decl[cur_func_name.getString()].push_back(cur_var_name); 
          }
        } else if (type->isFloatType()) {
          cout << "\tvar decl: " << (*i)->unparseToString() << "\n";
          has_fp_decl = true;
          fvar_decl[cur_func_name.getString()].push_back(cur_var_name); 
        } else if (isSgTypedefType(type) != NULL && isSgTypedefType(type)->get_name().getString() == "double3" ) {
          cout << "\tvar decl: " << (*i)->unparseToString() << "\n";
          has_fp_decl = true;
          fvar_decl[cur_func_name.getString()].push_back(cur_var_name); 
        }
        break;

        
        //printf("Error: not supported operations for record\n");
    }
  }
#ifdef _DEBUG
  // fill completed function profiling result
  if (cur_func_name != NULL)   {
        cout << "\t\tfunc name: " << cur_func_name.getString() << ", size: " << func_loop.size()<< "\n";
    int f_loop_size = func_loop[cur_func_name].size();
    // update profile info for out of loop operations
    for (int j=0; j<total_opt; j++) {
      loop_info[0].nbop += loop_info[0].nbop_cnt[j];
      func_loop[cur_func_name][0].nbop += func_loop[cur_func_name][0].nbop_cnt[j];
    }
    for (int j=0; j<f_loop_size; j++) {
      // get max0mum ad for a completed function based on the loop profiling result
      if (func_loop[cur_func_name][j].max_ad > func_map[cur_func_name].max_ad)
        func_map[cur_func_name].max_ad = func_loop[cur_func_name][j].max_ad;
      // get total nbop for a completed function
      func_map[cur_func_name].nbop += func_loop[cur_func_name][j].nbop;
    }
  }

  // sort tuning profiling info for loops
  tuning_sort(sfile);
#endif
}

int fpInstrument::tune_op(SgNode* node, SgBinaryOp* nop) {

  int ad = 0;
  SgBinaryOp* bop = nop;
  Rose_STL_Container<SgNode*> nodeList = node->get_traversalSuccessorContainer();
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    SgBinaryOp* iop = isSgBinaryOp(*i);
    if (iop == NULL) {
#ifdef _DEBUG
      cout << "\t\t node: " << (*i)->unparseToString() << ", none op\n";
#endif
      continue;
    }
    if (bop->variantT() == iop->variantT() || isSameOp(bop, iop)) {
      ad++;
      ad += tune_op(*i, nop);
      attachArbitraryText(isSgLocatedNode(*i), "to_double(", PreprocessingInfo::before);
      attachArbitraryText(isSgLocatedNode(*i), ")", PreprocessingInfo::after);
#ifdef _DEBUG
      cout << "\t\t node: " << (*i)->unparseToString() << ", same op\n";
#endif
    }
    if (bop->variantT() != iop->variantT() && !isSameOp(bop, iop)) {
      attachArbitraryText(isSgLocatedNode(*i), "to_double(", PreprocessingInfo::before);
      attachArbitraryText(isSgLocatedNode(*i), ")", PreprocessingInfo::after);
#ifdef _DEBUG
    cout << "\t\t node: " << (*i)->unparseToString() << ", diff op\n";
#endif
    }
  }
  return ad;
}

void fpInstrument::tuning_exe(SgForStatement *forstmt, int op_id) {
  #if 1
  SgStatement* forbodystmt = forstmt->get_loop_body(); 
  SgNode* bodynode = (SgNode*) forbodystmt;
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(bodynode, V_SgStatement);
  //Rose_STL_Container<SgNode*> nodeList = forbodystmt->get_traversalSuccessorContainer();
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++) {
    SgStatement* stmt = isSgStatement(*i);
    SgExprStatement* expr_stmt = isSgExprStatement(*i);
    SgExpression* expr;
    if (stmt == NULL)
      continue;
    if (stmt->variantT() == V_SgExprStatement) {
      Rose_STL_Container<SgNode*> boplist = NodeQuery::querySubTree((*i), V_SgBinaryOp);
      for (Rose_STL_Container<SgNode*>::iterator j = boplist.begin(); 
           j != boplist.end(); j++){
        //Rose_STL_Container<SgNode*> expList = (*j)->get_traversalSuccessorContainer();
        SgBinaryOp* bop = isSgBinaryOp(*j);
        #if 0
        if (bop->variantT() == V_SgAssignOp) {
          SgNode* rhs_op = bop->get_rhs_operand();
          SgBinaryOp* rhs_bop = (SgBinaryOp*) rhs_op;
          cout<< "\t\trhs expr: " << rhs_op->unparseToString() << "\n";
          if (isSgAddOp(rhs_op) || isSgAddOp(*j))
            cout<< "\ttune_add: " << rhs_op->unparseToString() << "\n";
            
        }
        #endif
        switch (bop->variantT())
        {
          case V_SgAssignOp:
            break;
          case V_SgPlusAssignOp:
          case V_SgMinusAssignOp:
          case V_SgAddOp:
          case V_SgSubtractOp:
            if (op_id == 0) {
              cout<< "\ttune_add: " << (*j)->unparseToString() << "\n";
              int skip;
              skip = tune_op(*j, bop);
              j+=skip;
              //i+=skip;
              //attachArbitraryText(isSgLocatedNode(*j), "to_double(", PreprocessingInfo::before);
            }
            break;
          case V_SgMultAssignOp:
          case V_SgMultiplyOp:
            if (op_id == 1) {
              cout<< "\ttune_mul: " << (*j)->unparseToString() << "\n";
              int skip;
              skip = tune_op(*j, bop);
              j+=skip;
            }
            break;
          case V_SgDivAssignOp:
          case V_SgDivideOp:
            if (op_id == 2) {
              cout<< "\ttune_div: " << (*j)->unparseToString() << "\n";
              int skip;
              skip = tune_op(*j, bop);
              j+=skip;
            }
            break;
        }
      }
    }
  }
  #endif

}

void fpInstrument::tuning_visit(SgSourceFile* sfile, int func_iter, int cur_iter, int tune_op) {

  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(sfile, V_SgStatement);
  // build outer loop profling container and insert it to loop_info
  int tuning_flag = 1;
  int tuning_order = 0;
  bool v_func = false;
  int func_id = 0;
  #if 0
  int loop_id = 0;
  int outloop_id = 0;
  loop_id = loop_info.size();
  assert(loop_id == outloop_id);
  fp_info init_info = {{0, 0, 0, 0}, {0, 0, 0, 0}, {3, 2, 1, 0}, 0, 0, 0};
  loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
  SgStatement* loop_cond;
  SgNode* loop_size_var;
  #endif

  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    
    // retrieve statement
    SgStatement* stmt = isSgStatement(*i);
    #if 1
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    #endif
    // check function
    // check current function
    SgFunctionDefinition* cur_func_def = isSgFunctionDefinition(*i); 
    if (cur_func_def != NULL) {
      if (func_map.find(cur_func_def->get_declaration()->get_name()) != func_map.end()) {
        v_func = true;
      }
    }
    if (!v_func)
      continue;

    #if 1
    if (stmt->variantT() == V_SgPragmaDeclaration) {
      SgPragmaDeclaration* pragma = (SgPragmaDeclaration*)(*i);
      string t_loop = pragma->get_pragma()->get_pragma();
      if (t_loop.find("function") != std::string::npos) {
        const char* t_ptr = t_loop.c_str();
        func_id = atoi(&t_ptr[9]);
        cout << " comment: " << t_loop << " , tuning order: " << func_id << "\n" ;
        tuning_flag = 1;
        continue;
      }
      if (tuning_flag == 1) {
        const char* t_ptr = t_loop.c_str();
        tuning_order = atoi(&t_ptr[0]);
        if (func_id == func_iter && tuning_order == cur_iter) { 
          cout << " current_tuning_loop " << t_loop << "\n" ;
        } else {
          cout << " tuning_loop " << pragma->get_pragma()->get_pragma() << ", order: " 
               << tuning_order << "\n" ;
        }
      }
    }
    #endif
    
    #if 0
    // check and tune corresponding loop
    if (tuning_flag == 1 && stmt->variantT() == V_SgForStatement && tuning_order == cur_iter) {
      tuning_exe(isSgForStatement(*i), tune_op);    
      tuning_order = 0;
    }
    #endif

    #if 0
    // loop condition statement
    int loop_level = 0;
    int num_skip_stmt = 0;
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    // check statement type to get loops or normal flops
    switch (stmt->variantT())
    {
      // statements for declaration, definition and flops
      case V_SgExprStatement:
        expr_profile(stmt, *i, outloop_id);
#ifdef _DEBUG
        cout << "expr node, " << "level: " << loop_level 
             << " stmt: " << (*i)->unparseToString() << "\n" ;
#endif
        break;
      // statements for for loops
      case V_SgForStatement:
        // loop count increment and assign loop id to new identified loop
        loop_level++;
        // get loop size variable 
        loop_cond = getLoopCondition(isSgScopeStatement(*i));
        loop_size_var = compare_profile(loop_cond->get_traversalSuccessorByIndex(0));
        // get loop_id for new identified loop
        loop_id = loop_info.size();
        loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
        cout << "for node, " << "id: " << loop_id <<", level: " 
             << loop_level << " size: " 
             << loop_size_var->unparseToString() << " stmt: " 
             << (*i)->unparseToString() << "\n" ;
        // profile loop information
        if (isSgForStatement(*i) != NULL) {
          num_skip_stmt += forstatement(isSgForStatement(*i), loop_level, loop_id);
#ifdef _DEBUG
          cout << "\t number of stmt: " << num_skip_stmt << "\n" ;
#endif
        }
        i += num_skip_stmt;
        break;
      default:
        //printf("Error: not supported operations for record\n");
        break;
    }
    #endif
  }
}

void fpInstrument::nestedloop_size(SgForStatement *forstmt, int level, int id) {
  int loop_id = 0;
  fp_info init_info = {{0, 0, 0, 0}, {0, 0, 0, 0}, {3, 2, 1, 0}, 0, 0, 0};
  // get loop body
  SgStatement* forbodystmt = forstmt->get_loop_body(); 
  SgNode* bodynode = (SgNode*) forbodystmt;
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(bodynode, V_SgStatement);
  int num_children = forbodystmt->get_numberOfTraversalSuccessors();
  int count = 0;
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++) {
    count++;
    SgStatement* stmt = isSgStatement(*i);
    SgExprStatement* expr_stmt = isSgExprStatement(*i);
    SgExpression* expr;
    if (stmt == NULL)
      continue;
    // check statement type to get loops or normal flops
    if (stmt->variantT() == V_SgForStatement) {
      //insert for into visited for container
      if (std::find(for_visited.begin(), for_visited.end(), *i) == for_visited.end())
        for_visited.push_back(*i);
      else
        continue;
      // loop level increment for nested loop
      level++;
      // get loop_id for new identified loop
      loop_id = loop_info.size();
      loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
      // insert counter variable for dynamic loop profiling
      string loop_var = "loop_counter" + Rose::StringUtility::numberToString(loop_id);
      SgVariableDeclaration* loop_count = buildVariableDeclaration (loop_var, buildIntType());
      insertStatementBefore(stmt, loop_count, false);
      // insert counter initialization
      string var_init = "loop_counter" + Rose::StringUtility::numberToString(loop_id) + "=0;";
      SgStatement* init_stmt = buildStatementFromString(var_init, getScope(*i));
      insertStatementBefore(stmt, init_stmt, false);
      if (isSgForStatement(*i) != NULL) {
        // get loop body
        SgStatement* forbodystmt = isSgForStatement(*i)->get_loop_body(); 
        SgNode* bodynode = (SgNode*) forbodystmt;
        // insert loop counter plusplus
        string cont_pp = "loop_counter" + Rose::StringUtility::numberToString(loop_id) + "++;";
        SgStatement* ppstmt = buildStatementFromString(cont_pp, getScope(*i));
        insertStatementBefore(isSgStatement(bodynode), ppstmt, false);
        // analysis loop
        cout << "\tfor node, " << "id, " << loop_id << " level: " 
             << level << " stmt: " << (*i)->unparseToString() << "\n" ;
        nestedloop_size(isSgForStatement(*i), level, loop_id);
      }
    }
  }
  // delete extra stmt from
}


void fpInstrument::loopsize_profile(SgSourceFile* sfile) {
  int loop_id = 0;
  fp_info init_info = {{0, 0, 0, 0}, {0, 0, 0, 0}, {3, 2, 1, 0}, 0, 0, 0};

  #if 1
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(sfile, V_SgStatement);

  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    // retrieve statement
    SgStatement* stmt = isSgStatement(*i);
    // loop condition statement
    int loop_level = 0;
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    // check current function 
    SgFunctionDefinition* cur_func_def = isSgFunctionDefinition(*i); 
    if (cur_func_def != NULL) {
      //cout << "\t\tStatement: " << (*i)->unparseToString() << "\n";
      if (func_map.find(cur_func_def->get_declaration()->get_name()) != func_map.end()) {
        cout << "\tfunc name: " << cur_func_def->get_declaration()->get_name().getString() << "\n";
      }
    }
      
    //if (stmt->variantT() == V_SgFunctionDefinition)
    //  cout << "\t\tStatement: " << (*i)->unparseToString() << "\n";
    // check statement type to get loops or normal flops
    #if 0
    SgExprStatement* func_call = isSgExprStatement(*i);
    if (func_call != NULL) {
      SgExpression* func_expr = func_call->get_expression();
      SgNode* func_node = (SgNode*) func_expr;
      if (isSgFunctionCallExp(func_node) != NULL && !func_expr->isLValue() && isSgFunctionCallExp(func_node)->isInMemoryPool()) {
        cout << "\t\tfunc call: " << (*i)->unparseToString() << "\n";
      }
    }
    #endif
    if (stmt->variantT() == V_SgForStatement) {

      //insert for into visited for container; if it exists, then go to next statement
      if (std::find(for_visited.begin(), for_visited.end(), *i) == for_visited.end())
        for_visited.push_back(*i);
      else
        continue;
      // get loop_id for new identified loop
      loop_id = loop_info.size();
      loop_info.insert(std::pair<int, fp_info>(loop_id, init_info));
      // insert counter variable for dynamic loop profiling
      string loop_var = "loop_counter" + Rose::StringUtility::numberToString(loop_id);
      SgVariableDeclaration* loop_count = buildVariableDeclaration (loop_var, buildIntType());
      insertStatementBefore(stmt, loop_count, false);
      // insert counter initialization
      string var_init = "loop_counter" + Rose::StringUtility::numberToString(loop_id) + "=0;";
      SgStatement* init_stmt = buildStatementFromString(var_init, getScope(*i));
      insertStatementBefore(stmt, init_stmt, false);
      if (isSgForStatement(*i) != NULL) {
        // get loop body
        SgStatement* forbodystmt = isSgForStatement(*i)->get_loop_body(); 
        SgNode* bodynode = (SgNode*) forbodystmt;
        // insert loop counter plusplus
        string cont_pp = "loop_counter" + Rose::StringUtility::numberToString(loop_id) + "++;";
        SgStatement* ppstmt = buildStatementFromString(cont_pp, getScope(*i));
        insertStatementBefore(isSgStatement(bodynode), ppstmt, false);
      }
      // loop count increment and assign loop id to new identified loop
      loop_level++;
      cout << "for node, " << "id: " << loop_id <<", level: " 
           << loop_level << " stmt: " 
           << (*i)->unparseToString() << "\n" ;
      // profile loop information
      if (isSgForStatement(*i) != NULL) {
        nestedloop_size(isSgForStatement(*i), loop_level, loop_id);
      }
    }
  }
  #endif

}

void fpInstrument::function_visit(SgSourceFile* sfile) {

  int func_size;

  //get the function list of input program
  string f_line;
  string open_name = ben_dir+"func_list.txt";
  cout << "Function list:" << open_name << "\n";
  ifstream func_file(open_name);
  if (func_file.is_open()) {
    while(getline(func_file, f_line)) {
      func_visit.push_back(f_line);
    }
    func_file.close();
  }
  cout << "Function should be visited: \n";
  for (std::vector<string>::iterator i = func_visit.begin(); 
       i != func_visit.end(); i++){
    cout << "\t" << *i << " " << (*i).size() << "\n";
  }
  cout << "\n\n";

  string open_host = ben_dir+"host_func.txt";
  ifstream h_func_file(open_host);
  if (h_func_file.is_open()) {
    while(getline(h_func_file, f_line)) {
      host_func.push_back(f_line);
    }
    h_func_file.close();
  }
  cout << "Host Function: \n";
  for (std::vector<string>::iterator i = host_func.begin(); 
       i != host_func.end(); i++){
    cout << "\t" << *i << " " << (*i).size() << "\n";
  }
  cout << "\n\n";

  string open_var = ben_dir+"device_vars.txt";
  ifstream var_file(open_var);
  if (var_file.is_open()) {
    while(getline(var_file, f_line)) {
      var_device.push_back(f_line);
    }
    func_file.close();
  }
  cout << "Device variabels should be noticed: \n";
  for (std::vector<string>::iterator i = var_device.begin(); 
       i != var_device.end(); i++){
    cout << "\t" << *i << " " << (*i).size() << "\n";
  }
  cout << "\n\n";

  //traverse the functions from the input program

  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(sfile, V_SgStatement);
  //get the function list of the program
  #if 0
  // find main
  SgFunctionDefinition* mainFunc = findMain(nodeList[0])->get_definition();
  // add main function itno function list
  func_size = func_map.size();
  func_info func_init = {func_size, 0, 0};
  func_map.insert(std::pair<SgName, func_info>(mainFunc->get_declaration()->get_qualified_name(), func_init));
  #endif
  SgName cur_func_name;
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){
    // retrieve statement
    SgStatement* stmt = isSgStatement(*i);
    // loop condition statement
    int loop_level = 0;
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;
    // check current function 
    SgFunctionDefinition* cur_func_def = isSgFunctionDefinition(*i); 
    if (cur_func_def != NULL) {
      cur_func_name = cur_func_def->get_declaration()->get_name();
      //cout << "\t\tStatement: " << cur_func_def->get_declaration()->get_name().getString() << "\n";
      if (std::find(func_visit.begin(), func_visit.end(), cur_func_name.getString()) != func_visit.end()) {
        cout << "\tcur func name: " << cur_func_def->get_declaration()->get_name().getString() << "\n";
        func_size = func_map.size();
        func_info func_init = {func_size, 0, 0};
        func_map.insert(std::pair<SgName, func_info>(cur_func_name, func_init));
      }
    }
  }
  #if 0
  cout << "\tmain func name: " << mainFunc->get_qualified_name().getString() << "\n";
  SgStatement* main_stmt = mainFunc->firstStatement();
  SgNode* main_node = (SgNode*) main_stmt;
  while (main_stmt != NULL) {
    main_node = (SgNode*) main_stmt;
    SgExprStatement* func_call = isSgExprStatement(main_node);
    //cout << "\t\tmain func call: " << main_node->unparseToString() << "stmt: " << main_stmt->variantT()<< "\n";
    if (func_call != NULL) {
      SgExpression* func_expr = func_call->get_expression();
      SgNode* func_node = (SgNode*) func_expr;
      //if (isSgFunctionCallExp(func_node) != NULL && !func_expr->isLValue()) {
      if (isSgFunctionCallExp(func_node) != NULL) {
     // if (isSgCudaKernelCallExp(main_node) != NULL) {
        cout << "\t\tmain func call: " << main_node->unparseToString() << ", traversal number: " << isSgFunctionCallExp(func_node)->getAssociatedFunctionDeclaration()->get_numberOfTraversalSuccessors() << "\n";
        func_size = func_map.size();
        func_info func_init = {func_size, 0, 0};
        func_map.insert(std::pair<SgName, func_info>(isSgFunctionCallExp(func_node)->getAssociatedFunctionDeclaration()->get_qualified_name(), func_init));
      }
    }
    main_stmt = getNextStatement(main_stmt);
  }
  #endif

  cout <<"\nfunction list size: " << func_map.size() << "\n";

}

void fpInstrument::recur_set(SgType* array_type, string type, SgScopeStatement* scope) {
  if (array_type->hasExplicitType())
    recur_set(array_type->dereference(), type, scope);
  else
    array_type->reset_base_type(buildOpaqueType(type, scope));

}
#define _DEBUG
void fpInstrument::TuneHigh(SgVariableDeclaration* decl, SgNode* node, int func_type, int var_type) {

  int var_case = 0;
  string var_string;
  SgInitializedName* def_name = decl->get_definition()->get_vardefn();
  string cur_var_name = def_name->get_name().getString();
  //SgInitializedName* init_name = decl->get_decl_item(def_name->get_name());
  SgInitializedName* init_name = getFirstInitializedName(decl);
  SgType* init_type;
  SgVariableDefinition* var_def = decl->get_definition(init_name);
  bool db_prt = false;

  switch (func_type) {
    case 1:
      // device variables in main function
      if (find(var_device.begin(), var_device.end(), init_name->get_name().getString()) 
          != var_device.end()) {
        if (var_type == 1) {
          //init_name->get_type()->get_ptr_to()->set_base_type(buildOpaqueType("gdd_real", getScope(node)));
          //init_name->get_type()->findBaseType()->reset_base_type(buildOpaqueType("gdd_real", getScope(node)));
          var_case = 11;
          var_string = "gdd_real";
          db_prt = true;
        }
        else if (var_type == 2) {
          init_name->get_type()->reset_base_type(buildOpaqueType("gdd_real3", getScope(node)));
          var_case = 21;
          var_string = "gdd_real3";
          db_prt = true;
        }
        else if (var_type == 3) {
          init_name->set_type(buildOpaqueType("gdd_real", getScope(node)));
          var_case = 11;
          var_string = "gdd_real";
        }
        else if (var_type == 4) {
          init_name->set_type(buildOpaqueType("gdd_real3", getScope(node)));
          var_case = 21;
          var_string = "gdd_real3";
        }
#ifdef _DEBUG
        cout << "\tdevice variable: " << def_name->get_name().getString() << "\n";
#endif
      } else {
        if (var_type == 1) {
          //recur_set(init_name->get_type(), "dd_real", getScope(node));
          //init_name->get_type()->get_ptr_to()->set_base_type(buildOpaqueType("dd_real", getScope(node)));
          var_case = 10;
          var_string = "dd_real";
        }
        else if (var_type == 2) {
          init_name->get_type()->reset_base_type(buildOpaqueType("dd_real3", getScope(node)));
          var_case = 20;
          var_string = "dd_real3";
        }
        else if (var_type == 3) {
          init_name->set_type(buildOpaqueType("dd_real", getScope(node)));
          var_case = 10;
          var_string = "dd_real";
        }
        else if (var_type == 4) {
          init_name->set_type(buildOpaqueType("dd_real3", getScope(node)));
          var_case = 20;
          var_string = "dd_real3";
        }
#ifdef _DEBUG
        cout << "\thost variable: " << def_name->get_name().getString() << "\n";
#endif
      }
      break;
    case 2:
      if (var_type == 1) {
        //init_name->get_type()->reset_base_type(buildOpaqueType("gdd_real", getScope(node)));
        var_case = 11;
        var_string = "gdd_real";
      }
      else if (var_type == 2) {
        //init_name->get_type()->reset_base_type(buildOpaqueType("gdd_real3", getScope(node)));
        var_case = 21;
        var_string = "gdd_real3";
      }
      else if (var_type == 3) {
        init_name->set_type(buildOpaqueType("gdd_real", getScope(node)));
        var_case = 11;
        var_string = "gdd_real";
      }
      else if (var_type == 4) {
        init_name->set_type(buildOpaqueType("gdd_real3", getScope(node)));
        var_case = 21;
        var_string = "gdd_real3";
      }
      break;
    case 3:
      if (var_type == 1) {
        //init_name->get_type()->reset_base_type(buildOpaqueType("dd_real", getScope(node)));
        var_case = 11;
        var_string = "dd_real";
      }
      else if (var_type == 2) {
        //init_name->get_type()->reset_base_type(buildOpaqueType("dd_real3", getScope(node)));
        var_case = 21;
        var_string = "dd_real3";
      }
      else if (var_type == 3) {
        init_name->set_type(buildOpaqueType("dd_real", getScope(node)));
        var_case = 11;
        var_string = "dd_real";
      }
      else if (var_type == 4) {
        init_name->set_type(buildOpaqueType("dd_real3", getScope(node)));
        var_case = 21;
        var_string = "dd_real3";
      }
      break;
    case 4:
      if (find(var_device.begin(), var_device.end(), init_name->get_name().getString()) 
          != var_device.end()) {
        if (var_type == 1) {
          //init_name->get_type()->reset_base_type(buildOpaqueType("__constant__ gdd_real", getScope(node)));
          var_case = 11;
          var_string = "gdd_real";
        }
        else if (var_type == 2) {
          //init_name->get_type()->reset_base_type(buildOpaqueType("__constant__ gdd_real3", getScope(node)));
          var_case = 21;
          var_string = "gdd_real3";
        }
        else if (var_type == 3) {
          init_name->set_type(buildOpaqueType("__constant__ gdd_real", getScope(node)));
          var_case = 11;
          var_string = "gdd_real";
        }
        else if (var_type == 4) {
          init_name->set_type(buildOpaqueType("__constant__ gdd_real3", getScope(node)));
          var_case = 21;
          var_string = "gdd_real3";
        }
      } else {
        var_case = 10;
        var_string = "dd_real";
      }
      //init_name->get_type()->reset_base_type(buildOpaqueType("dd_real", getScope(node)));
      break;
    case 5:
      break;
    default:
      break;
  }
  //change allocation and membercpy type 

  //change new type 
  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(node, V_SgNode);
  #if 1
  if (nodeList.size() > 0) {
    for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
         i != nodeList.end(); i++){
      if (isSgNewExp(*i) != NULL) {
        //isSgNewExp(*i)->get_specified_type()->reset_base_type(buildOpaqueType(var_string, getScope(node)));
#ifdef _DEBUG
        cout << "\t\t  new type: "  << var_string << ", "<<(*i)->unparseToString() << "\n";
#endif
      } 
#if 0
      else if(func_type == 5 && nodeList.size()>0) {
        if (isSgFunctionCallExp(*i) != NULL) {
          SgFunctionCallExp* alloc_func = isSgFunctionCallExp(*i);
          cout << "\t\t  alloca type: "  << var_string << ", "<< (*i)->unparseToString() << ", type: " << alloc_func->get_type()->class_name()<< "\n";
        } else if (isSgType(*i) != NULL) {
          cout << "\t\t  type: "  << var_string << ", "<< (*i)->unparseToString() << "\n";
        }
      }
#endif
    }
  }
  #endif

}
#undef _DEBUG

void fpInstrument::traverseInput(SgSourceFile* sfile) {
  // note: add a new variables
  //SgBasicBlock *block = isSgBasicBlock(node);
  //if (block != NULL) {
  //  SgVariableDeclaration *variableDeclaration = buildVariableDeclaration ("a_dd", buildOpaqueType("dd_real", getScope(node)));
  //  prependStatement(variableDeclaration, block);
  //}

  // Instruction types to check floating-point type and allocation
  SgVariableDeclaration* varsdeclaration;
  //SgNewExp* new_expr = isSgNewExp(node);
  //SgDeleteExp* delete_expr = isSgDeleteExp(node);
  //SgAllocateStatement*   varsallocation  = isSgAllocateStatement(node);
  SgType* def_type;
  SgTypedefType* tydef;
  SgFunctionDefinition* cur_func = NULL;
  SgInitializedNamePtrList arg_list;
  SgExpression* rhs_expr;
  SgType* type;
  SgType* ref_type;
  SgName cur_func_name;
  string cur_var_name;
  int func_type;

  Rose_STL_Container<SgNode*> nodeList = NodeQuery::querySubTree(sfile, V_SgStatement);
  for (Rose_STL_Container<SgNode*>::iterator i = nodeList.begin(); 
       i != nodeList.end(); i++){

    // retrieve statement
    SgStatement* stmt = isSgStatement(*i);
    // if it is from system header file, skip it
    if (insideSystemHeader(stmt))
      continue;
    // if it is an unexpected headerfile node, skip it
    if (headerfile_check(isSgLocatedNode(*i)))
      continue;

    //check the current function 
    if (isSgFunctionDefinition(*i) != NULL) {
      cur_func = isSgFunctionDefinition(*i); 
      //if (cur_func_name != cur_func->get_declaration()->get_name())
      //  cout << "func: "     << cur_func->get_declaration()->get_name().getString();
      //4: functions not visit, 1: main function, 2: device function, 3: global definition
      cur_func_name = cur_func->get_declaration()->get_name();  
      if (func_map.find(cur_func_name) != func_map.end()) { 
        cout << "device func: " << cur_func_name << "\n";
        func_type = 2;
      } else if (cur_func_name.getString() == "main") {
        cout << "func: " << cur_func_name << "\n";
        func_type = 1;
      } else if (std::find(host_func.begin(), host_func.end(), cur_func_name.getString()) != host_func.end()){
        cout << "host func: " << cur_func_name << "\n";
        func_type = 3;
      } else {
        func_type = 4;
      }
    }
    if (isSgFunctionDefinition(*i) != NULL) {
      cur_func = isSgFunctionDefinition(*i); 
      cur_func_name = cur_func->get_declaration()->get_name();  

      if (func_type == 2 || func_type == 3) {
        //maximize type in function arguments
        arg_list = cur_func->get_declaration()->get_args();
        for (int j=0; j<(arg_list.size()); j++) {
          SgInitializedName* def_name = arg_list[j];
          string cur_var_name = def_name->get_name().getString();
          tydef    = isSgTypedefType(def_name->get_type()->dereference());
          if (def_name->get_type()->hasExplicitType() ) {
            if (def_name->get_type()->dereference()->isFloatType()) {
#ifdef _DEBUG
              cout << "\t name: " << cur_var_name << "\n";
#endif
              //def_name->get_type()->reset_base_type(buildFloatType());
              if (func_type == 2) { 
                 def_name->get_type()->reset_base_type(buildOpaqueType("gdd_real", cur_func->get_declaration()->get_scope()));
                 cout << "\t device function.\n";
              } else { 
                 cout << "\t host function.\n";
                 def_name->get_type()->reset_base_type(buildOpaqueType("dd_real", cur_func->get_declaration()->get_scope()));
              }
            } else if(tydef != NULL && tydef->get_name().getString() == "double3" ) {
              if (func_type == 2) { 
                 cout << "\t device function.\n";
                def_name->get_type()->reset_base_type(buildOpaqueType("gdd_real3", cur_func->get_declaration()->get_scope()));
              } else {
                 cout << "\t host function.\n";
                def_name->get_type()->reset_base_type(buildOpaqueType("dd_real3", cur_func->get_declaration()->get_scope()));
              }
#ifdef _DEBUG
              cout << "\t db3* name: " << cur_var_name << "\n";
#endif
            }
          } else if (def_name->get_type()->isFloatType()) {
              if (func_type == 2)  {
                 cout << "\t device function.\n";
                def_name->set_type(buildOpaqueType("gdd_real", cur_func->get_declaration()->get_scope()));
              } else {
                 cout << "\t host function.\n";
                def_name->set_type(buildOpaqueType("dd_real", cur_func->get_declaration()->get_scope()));
              }
#ifdef _DEBUG
              cout << "\t dbname: " << cur_var_name << "\n";
#endif
          } else if (isSgTypedefType(def_name->get_type()) != NULL && isSgTypedefType(def_name->get_type())->get_name().getString() == "double3") {
              if (func_type == 2) { 
                 cout << "\t device function.\n";
                def_name->set_type(buildOpaqueType("gdd_real3", cur_func->get_declaration()->get_scope()));
              } else {
                 cout << "\t host function.\n";
                def_name->set_type(buildOpaqueType("dd_real3", cur_func->get_declaration()->get_scope()));
              }
#ifdef _DEBUG
              cout << "\t db3 name: " << cur_var_name << "\n";
#endif
          }
        }
        //maximize the function type
        type = cur_func->get_declaration()->get_type()->get_return_type(); 
        if (type->hasExplicitType() ) {
          tydef    = isSgTypedefType(type->dereference());
          if (type->dereference()->isFloatType()) {
            //def_name->get_type()->reset_base_type(buildFloatType());
            if (func_type == 2) { 
                 cout << "\t device function.\n";
              type->reset_base_type(buildOpaqueType("gdd_real", cur_func->get_declaration()->get_scope()));
            } else {
                 cout << "\t host function.\n";
              type->reset_base_type(buildOpaqueType("dd_real", cur_func->get_declaration()->get_scope()));
            }
          } else if(tydef != NULL && tydef->get_name().getString() == "double3" ) {
            if (func_type == 2) { 
                 cout << "\t device function.\n";
              type->reset_base_type(buildOpaqueType("gdd_real3", cur_func->get_declaration()->get_scope()));
            } else {
                 cout << "\t host function.\n";
              type->reset_base_type(buildOpaqueType("dd_real3", cur_func->get_declaration()->get_scope()));
            }
#ifdef _DEBUG
            cout << "\t db3* return: " << cur_var_name << "\n";
#endif
          }
        } else if (type->isFloatType()) {
           //cur_func->get_declaration()->get_type()->set_return_type(buildOpaqueType("gdd_real", cur_func->get_declaration()->get_scope()));
           if (func_type == 2) { 
                 cout << "\t device function.\n";
             cur_func->get_declaration()->set_type(new SgFunctionType(buildOpaqueType("gdd_real", cur_func->get_declaration()->get_scope()),true));
           } else {
                 cout << "\t host function.\n";
             cur_func->get_declaration()->set_type(new SgFunctionType(buildOpaqueType("dd_real", cur_func->get_declaration()->get_scope()),true));
            }
#ifdef _DEBUG
           cout << "\t db return: " << cur_var_name << "\n";
#endif
        } else if (isSgTypedefType(type) != NULL && isSgTypedefType(type)->get_name().getString() == "double3") {
            if (func_type == 2) { 
                 cout << "\t device function.\n";
              cur_func->get_declaration()->set_type(new SgFunctionType(buildOpaqueType("gdd_real3", cur_func->get_declaration()->get_scope()),true));
            } else {
                 cout << "\t host function.\n";
              cur_func->get_declaration()->set_type(new SgFunctionType(buildOpaqueType("dd_real3", cur_func->get_declaration()->get_scope()),true));
            }
#ifdef _DEBUG
            cout << "\t db3 return: " << cur_var_name << "\n";
#endif
        }
      }
    }

    varsdeclaration = isSgVariableDeclaration(*i);
    SgInitializedName* def_name;
    string cur_var_name;
    switch (stmt->variantT())
    {
      case V_SgVariableDeclaration:
        type =  varsdeclaration->get_definition()->get_type();
        def_name = varsdeclaration->get_definition()->get_vardefn();
        cur_var_name = def_name->get_name().getString();
        // check if it is a pointer or typedef type
        if (type->hasExplicitType()) {
          //def_type = type->dereference();
          def_type = type->findBaseType();
          tydef    = isSgTypedefType(def_type);
          // The double pointer type
          //if (def_type->isFloatType() || get_name(def_type) == "gdd_real" || get_name(def_type) == "dd_real") {
          if (def_type->isFloatType()) {
            TuneHigh(varsdeclaration, *i, func_type, 1);
            cout << "\tdouble pointer: " << (*i)->unparseToString() << ", type: " << func_type <<"\n";
          // The double3 pointer type
          } else if(tydef != NULL && tydef->get_name().getString() == "double3" ) {
            TuneHigh(varsdeclaration, *i, func_type, 2);
#ifdef _DEBUG
            cout << "\tdouble3 pointer: " << (*i)->unparseToString() << ", type: " << func_type <<"\n";
#endif
          } else {
            if (find(var_device.begin(), var_device.end(), cur_var_name) != var_device.end()) {
              TuneHigh(varsdeclaration, *i, 5, 0);
              cout << "\tmaximized double pointer: " << (*i)->unparseToString() << ", type: " << func_type <<"\n";
            }
          }
        //} else if (type->isFloatType()|| get_name(type) == "gdd_real" || get_name(type) == "dd_real") {
        } else if (type->isFloatType()) {
          TuneHigh(varsdeclaration, *i, func_type, 3);
#ifdef _DEBUG
          cout << "\tdouble: " << (*i)->unparseToString() << ", type: " << func_type <<"\n";
#endif
        } else if (isSgTypedefType(type) != NULL && isSgTypedefType(type)->get_name().getString() == "double3" ) {
          TuneHigh(varsdeclaration, *i, func_type, 4);
#ifdef _DEBUG
          cout << "\tdouble3: " << (*i)->unparseToString() << ", type: " << func_type <<"\n";
#endif
        }
        //if (func_type == 1) {
        //  ref_type =  varsdeclaration->get_definition()->get_type()->dereference();
        //  cout << "new : " << (*i)->unparseToString() << ", type: " << get_name(ref_type)  <<"\n";
        //}
        break;
      default:
        break;
    }
  }

}


int main (int argc, char** argv)
{
    // Build the AST used by ROSE
    SgProject* project = frontend(argc, argv);

    // Run internal consistency tests on AST
// AstTests::runAllTests(project);
// declaration parsing with Parser builing blocks
    int i, t_iter = 0, f_iter = 9, l_iter = 0;
    int tune_op = 0;
    //get tuning info
    string f_line;
    string bench_dir;
    ifstream tune_file("tune_info.txt");
    if (tune_file.is_open()) {
      for (i=0; i<2; i++) {
      //while(getline(tune_file, f_line)) {
        getline(tune_file, f_line);
        if (i==0)
          t_iter = stoi(f_line); 
        else
          bench_dir = f_line; 
      }
      tune_file.close();
    }
    // retrieve current tuning iteration
    //for (i=0; i<argc; i++) {
    //  if (argv[i] == "-t")
    //    t_iter = atoi(argv[i+1]);
    //}
    cout << "Current Tuning Iteration: " << t_iter << "\n";
    cout << "Current Tuning benchmark: " << bench_dir << "\n";
    // instrument program
    SgFilePtrList file_list = project->get_files();
    SgFilePtrList::iterator iter;
    if (file_list.size()>0) {
      cout <<"\nPorject has available files: " << file_list.size() << "\n";
      for (iter=file_list.begin(); iter!=file_list.end(); iter++) {
        SgFile* cur_file = *iter;
        SgSourceFile *sfile = isSgSourceFile(cur_file);
        if (sfile != NULL) {
          cout <<"files is valid. \n";
          fpInstrument fpinstrument;
          fpinstrument.set_dir(bench_dir);
          if (t_iter > 0) {
            cout << "\t tuning start: \n";
            fpinstrument.function_visit(sfile);
            fpinstrument.tuning_visit(sfile, f_iter, l_iter, tune_op);
            //fpinstrument.print_fp_operation_count();
            // Merge the testing code from Virtual machine
            // Generate dot file to view the AST Tree
            //AstDOTGeneration dotgen;
            //dotgen.generateInputFiles(project, AstDOTGeneration::PREORDER);
          } else if (t_iter < 0) {
            cout << "\t dynamic profiling start: \n";
            fpinstrument.function_visit(sfile);
            fpinstrument.traverseInput(sfile);
            //fpinstrument.loopsize_profile(sfile);
          } else {
            cout << "\t profiling start: \n";
            fpinstrument.function_visit(sfile);
            fpinstrument.visit(sfile);
            fpinstrument.print_fp_operation_count();
          }
        }
      }
    }
    // Generate source code from AST and invoke your
    // desired backend compiler
    cout << endl << endl << endl << endl << endl << endl ;
    return backend(project);
}


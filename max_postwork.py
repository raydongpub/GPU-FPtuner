import sys
import fileinput
import subprocess
from contextlib import closing
from collections import OrderedDict
import re

"""""""""""""""""""""""""""
" utility function section
"""""""""""""""""""""""""""
def alloc_size(stmt):
  size_list = stmt[stmt.find('[')+len('['):stmt.rfind(']')].replace("[", " ").replace("]", " ").split(" ")
  size_val = ""
  counter = 0
  for size in size_list:
    if not size.strip():
      continue
    if counter == 0:
      size_val += size 
    else:
      size_val += "*"+size 
    counter+=1
  return size_val  

def max_pattern(var_list, host_vars, db_vars, fl_vars, sd_vars, h_new, n_size, n_ptr, stmt_line):
  ret = 0
  #check device variable list
  for var in var_list:
    stmt_var = var.replace("\n", "")
    if var in stmt_line:
      if "double" in stmt_line or "dd_real" in stmt_line:
        if "=" in stmt_line and ("double" in stmt_line.split("=")[0] or "dd_real" in stmt_line) and var in stmt_line.split("=")[0]:
          cur_lhs = stmt_line.split("=")[0]
          if "dd_real" not in stmt_line:
            cur_lhs = cur_lhs.replace("double", "gdd_real")
            print(stmt_line.replace(stmt_line.split("=")[0], cur_lhs))
            ret = 11
            break
          elif "gdd_real" not in stmt_line:
            cur_lhs = cur_lhs.replace("dd_real", "gdd_real")
            print(stmt_line.replace(stmt_line.split("=")[0], cur_lhs))
            ret = 11
            break
          elif "alloc<" in stmt_line: 
              print(stmt_line.replace("double", "gdd_real"))
              ret = 11
              break
        elif "=" in stmt_line and "alloc<" in stmt_line: 
            print(stmt_line.replace("double", "gdd_real"))
            ret = 11
            break
        elif "=" not in stmt_line:
           if var in sd_vars and "__shared__" not in stmt_line and ";" in stmt_line:
               if "gdd_real" not in stmt_line:
                 print(stmt_line.replace('dd_real', '__shared__ gdd_real').replace('double', '__shared__ gdd_real'))
                 ret = 11
                 break   
               else:
                 print(stmt_line.replace('gdd_real', '__shared__ gdd_real'))
                 ret = 11
                 break   
           elif var not in sd_vars:
             ret = 1
             break   
        else:
          ret = 0

  #check host variable list
  for var in host_vars: 
    h_var = var.replace("\n", "")
    #if h_var in stmt_line:
    #if if_var_decl(h_var, stmt_line) and "main" not in stmt_line and "set_constants" not in stmt_line:
    if if_var_decl(h_var, stmt_line) and "main" not in stmt_line:
      if ( ("cudaMemcpy" in stmt_line or "upload" in stmt_line) and ret == 1):
        if (h_var in h_new):
          if "HostToDevice" in stmt_line or "upload" in stmt_line or "cudaMemcpyToSymbol" in stmt_line:
              nh_var = "n"+h_var
              #if h_var == "hm":
              #  print "var--- " + n_size[h_var]
              print(h_new[h_var])
              #insert qd2gqd() function 
              if h_var in n_size:
                if n_ptr[h_var]== 1:
                  qd2gqd = "qd2gqd("+h_var+","+nh_var+","+str(n_size[h_var])+");\n"
                elif n_ptr[h_var]==2:
                  size_set = n_size[h_var].replace("*", " ").split(" ")
                  qd2gqd = "qd2gqd2("+h_var+","+nh_var+"," + str(size_set[0]) + "," + str(size_set[1]) + ","+str(n_size[h_var])+");\n"
                else:
                  qd2gqd = "qd2gqd(&"+h_var+",&"+nh_var+","+str(n_size[h_var])+");\n"
                print(qd2gqd)
              if "cudaMemcpyToSymbol" in stmt_line:
                print(stmt_line.replace('double', 'gdd_real').replace(h_var, nh_var).replace("Symbol((&", "Symbol(("))
              else:
                print(stmt_line.replace('double', 'gdd_real').replace(h_var, nh_var))
              ret = 11
          elif "DeviceToHost" in stmt_line:
              nh_var = "n"+h_var
              if "cudaMemcpyToSymbol" in stmt_line:
                print(stmt_line.replace('double', 'gdd_real').replace(h_var, nh_var).replace("Symbol((&", "Symbol(("))
              else:
                print(stmt_line.replace('double', 'gdd_real').replace(h_var, nh_var))
              if h_var in n_size:
                if n_ptr[h_var]== 1:
                  qd2gqd = "gqd2qd("+nh_var+","+h_var+","+str(n_size[h_var])+");\n"
                elif n_ptr[h_var]==2:
                  size_set = n_size[h_var].replace("*", " ").split(" ")
                  qd2gqd = "gqd2qd2("+nh_var+","+h_var+"," + str(size_set[0]) + "," + str(size_set[1]) + ","+str(n_size[h_var])+");\n"
                else:
                  qd2gqd = "gqd2qd(&"+nh_var+",&"+h_var+","+str(n_size[h_var])+");\n"
                print(qd2gqd)
              ret = 11
        else:
          if "cudaMemcpyToSymbol" in stmt_line:
            print(stmt_line.replace('double', 'gdd_real').replace("Symbol((&", "Symbol(("))
            ret = 11
      elif "double" in stmt_line:
        if "=" in stmt_line and "double" in stmt_line.split("=")[0]:
          ret = 2
        elif "=" not in stmt_line and "double" in stmt_line:
          ret = 2
        else:
          ret = 0
      else:
        ret = 0
      if "double" in stmt_line or "dd_real" in stmt_line:
        if ("double" in stmt_line or "gdd_real" in stmt_line) and "new" in stmt_line:
          ret = 2
        if h_new.get(h_var) == None:
          #if h_var == "m":
          #  print "cur--- " + stmt_line
          #get the allocation stmt and tranform to gdd_real type
          nh_var = "n"+h_var
          nh_decl = None
          if "=" in stmt_line and "new" not in stmt_line:
            nh_decl = stmt_line.split('=')[0] + ";" 
          else:
            nh_decl = stmt_line
          if "gdd_real" in stmt_line :
            h_new[h_var] = nh_decl.replace('double', "gdd_real").replace(h_var, nh_var)
          elif "gdd_real" not in stmt_line:
            h_new[h_var] = nh_decl.replace('dd_real', "gdd_real").replace('double', "gdd_real").replace(h_var, nh_var)
          else:
            h_new[h_var] = nh_decl.replace("dd_real", "gdd_real").replace('double', "gdd_real").replace(h_var, nh_var)
          #get the size of the host variable 
          if "[" in stmt_line and "]" in stmt_line:
            #n_size[h_var] = stmt_line[stmt_line.find('[')+len('['):stmt_line.rfind(']')];
            if stmt_line.count('[') > 1:
              n_size[h_var] = alloc_size(stmt_line)
            else:
              n_size[h_var] = stmt_line[stmt_line.find('[')+len('['):stmt_line.rfind(']')];
            n_ptr[h_var] = stmt_line.count('[');
          else:
            n_size[h_var] = "1"
            n_ptr[h_var] = 0;
  #check double variable list
  for var in db_vars:
    d_var = var.replace("\n", "")
    if d_var in stmt_line:
      if "dd_real" in stmt_line or "double" in stmt_line:
        ret = 3
  for var in fl_vars:
    f_var = var.replace("\n", "")
    if f_var in stmt_line:
      if "dd_real" in stmt_line or "double" in stmt_line:
        ret = 4
  if ret == 0 and "double" in stmt_line and "(double )" not in stmt_line:
    if "=" in stmt_line and "double" in stmt_line.split("=")[0]:
      ret = 1
    elif "=" not in stmt_line and "double" in stmt_line:
      ret = 1
      
  return ret

def is_Ftype (l_val):
  if "double" in l_val:
    return False
  elif "int" in l_val:
    return False
  elif "dd_real" in l_val:
    return False
  elif "long" in l_val:
    return False
  elif "unsigned" in l_val:
    return False
  return True

def is_host(host_list, stmt_line):
  for host in host_list:
    host_var = host.replace("\n", "")
    if host_var in stmt_line:
      return True
  return False
def is_dev(dev_list, stmt_line):
  for dev in dev_list:
    dev_var = dev.replace("\n", "")
    if dev_var in stmt_line:
      return True
  return False
def is_global(dev_list, stmt_line):
  for dev in dev_list:
    if if_var_decl(dev, stmt_line):
      return True
  return False

def is_math(name):
  if name == "sqrt":
    return True;
  elif name == "exp":
    return True;
  return False

def exclude_func(stmt):
  if "dim3" in stmt:
    return False
  if "cudaError_t" in stmt:
    return False
  if "fprint" in stmt:
    return False
  if "exit" in stmt:
    return False
  if "upload" in stmt:
    return False
  if "alloc" in stmt:
    return False
  if "dealloc" in stmt:
    return False
  if "copy" in stmt:
    return False
  return True
def is_func(stmt):
  params = stmt[stmt.find('(')+len('('):stmt.rfind(')')]
  if "," in params and exclude_func(stmt):
    return True;
  if "+" in params:
    return False
  if "-" in params:
    return False
  if "*" in params:
    return False
  if "/" in params:
    return False
  if exclude_func(stmt)==False:
    return False
  return True

def has_op(line):
  if "+" in line:
    return False
  elif "-" in line:
    return False
  elif "*" in line:
    return False
  elif "/" in line:
    return False
  return True

def is_gdd(h_vars, stmt_line):
  #for var in gdd_vars:
  #  g_var = var.replace("\n", "");
  #  if g_var in stmt_line and "=" in stmt_line and has_op(stmt_line):
  #    return True
  #return False
  if "=" in stmt_line:
    for h_v in h_vars:
      if if_var_decl(h_v, stmt_line):
        return False
    cur_rhs = stmt_line.split("=")[1].replace(";", "")
    if "(double )" in cur_rhs:
      cur_rhs = cur_rhs.replace("(double )", "").replace("(", "").replace(")", "").replace(" ", "").replace("\n", "")
      if cur_rhs == "1.0" or cur_rhs == "0.0" or cur_rhs == "80.0":
        return True
  return False

def gdd_assign(host_vars, stmt_line):
  cur_lhs = stmt_line.split("=")[0].replace("double", "").replace("double3", "").replace("gdd_real", "").replace("gdd_real3", "")
  g_var = None
  if "[" in cur_lhs:
    re = cur_lhs[cur_lhs.find('[')+len('['):cur_lhs.rfind(']')]
    g_var = cur_lhs.replace(re, "").replace("[", "").replace("]", "").replace(" ", "")
  else:
    g_var = cur_lhs.replace(" ", "")

  if g_var not in host_vars:
    g_var = var.replace("\n", "").replace(" ", "");
    #if g_var in stmt_line and "=" in stmt_line and has_op(stmt_line):
      #if "gdd_real" in stmt_line:
    value = stmt_line[stmt_line.find('=')+len('='):stmt_line.rfind(';')]
        #l_val = stmt_line.split('=')[0].replace("gdd_real", "").replace(" ", "").replace(":", "").replace(";", "").replace("gdd_real3", "")
        #if (g_var != l_val):
        #  continue
    mod_value = "dd_add(" + value+", 0.0)"
    print(stmt_line.replace(value, mod_value))
    #  else:
    #    value = stmt_line[stmt_line.find('=')+len('='):stmt_line.rfind(';')]
    #    l_val = stmt_line.split('=')[0].replace(" ", "").replace(":", "").replace(";", "")
    #    if "." in l_val:
    #      l_val = l_val.split('.')[0]
    #    if (g_var != l_val):
    #      continue
    #    mod_value = "dd_add(" + value+", 0.0)"
    #    print(stmt_line.replace(value, mod_value))
#copy and replace the calling function for maximized program
def find_func(call_list, stmt):
  for f_body in call_list:
    if f_body == stmt.split('(')[0].split(' ')[-1]:
      return True 
  return False
def dup_func(stmt_func, func_body, call_ct):
  func_name = stmt_func.split('(')[0].split(' ')[-1].replace("\n", "")
  if func_name in call_ct:
    call_num = call_ct[func_name]
    for x in range(1, call_num):
      for line in func_body:
        if (func_name in line):
          print(line.replace(func_name, func_name+"_"+str(x)))
        else:
          print line
def find_call(call_list, call_id, stmt):
  for call in call_list:
    if call == stmt.split('(')[0].split(' ')[-1]:
      if call_id[call] == 0:
        call_id[call] += 1
        return stmt
      else:
        stmt = stmt.replace(call, (call+"_"+str(call_id[call])))
        call_id[call] += 1
        return stmt
  return stmt
def check_inside_func(func_call, n_call):
  for stmt_line in func_call:
    if not stmt_line.strip():
      continue
    if is_func(stmt_line) and ";" in stmt_line: 
      if "<<<" in stmt_line: 
        func_name = stmt_line.split('<<<')[0].split(' ')[-1]
      else:
        func_name = stmt_line.split('(')[0].split(' ')[-1]
      recur_func(func_name, n_call)

def recur_func(call, new_call):
  with open(sys.argv[2], "r") as stmt_list:
    inside_func = False
    f_cnt = 0
    func_call = []
    for stmt in stmt_list:
      if not stmt.strip():
        continue
      if call == stmt.split('(')[0].split(' ')[-1] and not ";" in stmt:
        inside_func = True
        if "{" in stmt:
          f_cnt += 1
        func_call.append(stmt)
      elif inside_func:
        if "{" in stmt:
          f_cnt += 1
        elif "}" in stmt:
          f_cnt -= 1
        #add stmt into target function call list
        func_call.append(stmt)
        #function load complete, modification begin
        if f_cnt == 0:
          inside_func = False
          check_inside_func(func_call, new_call)
          new_call.append(call)

def find_param_func(index, call):
  with open(sys.argv[2], "r") as stmt_list:
    for stmt in stmt_list:
      if not stmt.strip():
        continue
      if call == stmt.split('(')[0].split(' ')[-1] and not ";" in stmt:
        cur_p = stmt[stmt.find('(')+len('('):stmt.rfind(')')].split(',')[index]
        if "gdd_real" in cur_p:
          return cur_p.replace("gdd_real3", " ").replace("gdd_real", " ").replace("double3", " ").replace("double", " ").replace("&", " ").replace("*", " ").replace(" ", "").replace("\n", "")
        else:
          return cur_p.replace("gdd_real3", " ").replace("gdd_real", " ").replace("dd_real3", " ").replace("dd_real", " ").replace("double3", " ").replace("double", " ").replace("&", " ").replace("*", " ").replace(" ", "").replace("\n", "")

def call_inside(func_call, f_list, target_var):
  for stmt_line in func_call:
    if not stmt_line.strip():
      continue
    if ");" in stmt_line and is_func(stmt_line)==True and target_var in stmt_line:
      if "<<<" in stmt_line:
        params = stmt_line[stmt_line.find('>>>(')+len('>>>('):stmt_line.rfind(')')].split(',')
      else:
        params = stmt_line[stmt_line.find('(')+len('('):stmt_line.rfind(')')].split(',')
      clean_var(params)
      if "<<<" in stmt_line: 
        func_name = stmt_line.split('<<<')[0].split(' ')[-1]
      else:
        func_name = stmt_line.split('(')[0].split(' ')[-1]
      if target_var in params and "qd2gqd" not in stmt_line: 
        seen = set()
        for idx, item in enumerate(params):
          if item not in seen:
            seen.add(item)
            if func_name not in f_list and target_var==item:
              f_list[func_name] = [idx]
          else:
            if target_var==item:
              f_list[func_name].append(idx)
        #print "param func: " + func_name + " " + str(f_list)
        #print find_param_func(f_list[func_name][0], func_name)
        recur_param(func_name, f_list, find_param_func(f_list[func_name][0], func_name))
def recur_param(call, param_list, target_var):
  with open(sys.argv[2], "r") as stmt_list:
    inside_func = False
    f_cnt = 0
    func_call = []
    for stmt in stmt_list:
      if not stmt.strip():
        continue
      fc_name = None 
      if "<<<" in stmt:
        fc_name =  stmt.split('(')[0].split(' ')[-1]
      else:
        fc_name = stmt.split('(')[0].split(' ')[-1]
      if call == fc_name and not ";" in stmt:
        inside_func = True
        if "{" in stmt:
          f_cnt += 1
        func_call.append(stmt)
      elif inside_func:
        if "{" in stmt:
          f_cnt += 1
        elif "}" in stmt:
          f_cnt -= 1
        #add stmt into target function call list
        func_call.append(stmt)
        #function load complete, modification begin
        if f_cnt == 0:
          inside_func = False
          call_inside(func_call, param_list, target_var)

def if_target(ret_f, param_f, cur_f, stmt):
  #find the target function that assign value to the modified variables
  for ret in ret_f:
    ret = ret.replace("\n", "")
    if not ret.strip():
      continue
    if ret == stmt.split('(')[0].split(' ')[-1] and ";" not in stmt:
      return True
  #find the target function that use the modified variables as parameters
  for param in param_f:
    param = param.replace("\n", "")
    if not param.strip():
      continue
    if param == stmt.split('(')[0].split(' ')[-1] and ";" not in stmt:
      return True
  #find the target function that contains the expected variables
  if cur_f == stmt.split('(')[0].split(' ')[-1]:
    return True
  return False

def find_ret(ret_f, stmt):
  for ret in ret_f:
    ret = ret.replace("\n", "")
    if ret == stmt.split('(')[0].split(' ')[-1] and ";" not in stmt:
      return True
  return False
def change_ret(stmt):
  mod_decl = stmt.split('(')[0]
  if "gdd_real" in stmt:
    mod_decl = mod_decl.replace("gdd_real", "double")
  elif "gdd_real" not in stmt and "dd_real" in stmt:
    mod_decl = mod_decl.replace("dd_real", "double")
  elif "dd_real" not in stmt and "double" in stmt:
    mod_decl = mod_decl.replace("double", "float")
  stmt = stmt.replace(stmt.split('(')[0], mod_decl)
  return stmt

def find_return(gdd_decl, stmt):
  if "return" in stmt and "to_double" not in stmt:
    for gdd in gdd_decl:
      if gdd in stmt:
        ret_stmt = stmt[stmt.find('return ')+len('return '):stmt.rfind(';')] 
        stmt = stmt.replace(ret_stmt, ("to_double("+ret_stmt+")"))
        return stmt
  if "return " in stmt and "float" not in stmt and "return ;" not in stmt:
    ret_stmt = stmt[stmt.find('return ')+len('return '):stmt.rfind(';')] 
    stmt = stmt.replace(ret_stmt, ("(float)"+ret_stmt))
    return stmt
  return stmt
def find_param(param_f, stmt):
  for param in param_f:
    param = param.replace("\n", "")
    if param == stmt.split('(')[0].split(' ')[-1] and ";" not in stmt:
      return param_f[param]
  return -1
def change_param(index, stmt, is_fl):
  idx = int(index)
  cur_p = stmt[stmt.find('(')+len('('):stmt.rfind(')')].split(',')[idx]
  if "gdd_real" in cur_p:
    mod_param = cur_p.replace("gdd_real", "double")
    stmt = stmt.replace(cur_p, mod_param)
  elif "gdd_real" not in cur_p and "dd_real" in cur_p:
    mod_param = cur_p.replace("dd_real", "double")
    stmt = stmt.replace(cur_p, mod_param)
  #elif "dd_real" not in cur_p and "double" in cur_p and is_fl:
  elif "dd_real" not in cur_p and "double" in cur_p:
    mod_param = cur_p.replace("double", "float")
    stmt = stmt.replace(cur_p, mod_param)
  return stmt
def get_param(index, stmt):
  idx = int(index)
  cur_p = stmt[stmt.find('(')+len('('):stmt.rfind(')')].split(',')[idx]
  if "gdd_real" in cur_p:
    return cur_p.replace("gdd_real3", "").replace("gdd_real", "").replace("&", "").replace("*", "").replace(" ", "").replace("\n", "")
  elif "gdd_real" not in cur_p and "dd_real" in cur_p:
    return cur_p.replace("dd_real3", "").replace("dd_real", "").replace("&", "").replace("*", "").replace(" ", "").replace("\n", "")
  elif "dd_real" not in cur_p and "double" in cur_p:
    return cur_p.replace("double3", "").replace("double", "").replace("&", "").replace("*", "").replace(" ", "").replace("\n", "")
  elif "dd_real" not in cur_p and "float" in cur_p:
    return cur_p.replace("float3", "").replace("float", "").replace("&", "").replace("*", "").replace(" ", "").replace("\n", "")

def if_gdd(rhs, gdd_decl, db_decl):
  #hs_list = ''.join([i for i in rhs if not i.isdigit()]) 
  hs_list = rhs 
  if "[" in hs_list:
    tod = re.findall(r"\[(.*?)\]", hs_list)
    for var in tod:
      hs_list = hs_list.replace("["+var+"]", "")

  hs = hs_list.replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").replace("=", " ").replace("(", " ").replace(")", " ").replace("to_double", " ").replace("double3", " ").replace("double", " ").replace("::", " ").replace("gdd_real3", " ").replace ("gdd_real", " ").replace("dd_real3", " ").replace("dd_real", " ").replace(".", " ").replace(",", " ").replace("&", " ").replace("sqrt", " ").replace("exp", " ").replace("[", " ").replace("]", " ").replace("\n", "").split(" ")
  hs = filter(None, hs)
  #if "t1" in rhs or "bt" in rhs:
  #  print "ifgdd--- " + str(hs)
  for gdd in gdd_decl:
    if gdd in hs:
      return True
  return False
 
def balance_type(p_var, gdd_decl, db_decl, fl_decl, stmt):
  if p_var in stmt.split('=')[0] and "to_double" not in stmt:
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    cur_lhs = stmt.split('=')[0] 
    if "float" in cur_lhs and "(float)" not in stmt and if_gdd(cur_rhs, db_decl, fl_decl):
      mod_rhs = "(float)" + cur_rhs
      stmt = stmt.replace(cur_rhs, mod_rhs)
      return stmt

    #if p_var == "t1":
    #  print "cur--- " + stmt;
    #  print cur_rhs
    #  print cur_lhs
    #  print str(gdd_decl)
    if if_gdd(cur_rhs, gdd_decl, db_decl) and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
      mod_rhs = "to_double(" + cur_rhs + ")"
      stmt = stmt.replace(cur_rhs, mod_rhs)
      return stmt
  if p_var in stmt.split('=')[0] and "dd_add" in stmt:
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    cur_lhs = stmt.split('=')[0] 
    if if_gdd(cur_lhs, gdd_decl, db_decl)==False:
      mod_rhs = cur_rhs.replace("dd_add(", "").replace(", 0.0)", "")
      stmt = stmt.replace(cur_rhs, mod_rhs)
      return stmt
  if p_var in stmt and p_var not in stmt.split('=')[0] and "to_double" in stmt:
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    if if_gdd(cur_rhs, gdd_decl, db_decl)== False:
      return stmt.replace("to_double", "")
  if p_var in stmt and p_var not in stmt.split('=')[0] and "gdd_real" in stmt:
    cur_lhs = stmt.split('=')[0] 
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    #if if_gdd(cur_rhs, gdd_decl, db_decl)==False and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
    if if_gdd(cur_rhs, gdd_decl, db_decl)==False: #and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
      c_v = stmt.split('=')[0].replace("::gdd_real3", "").replace("::gdd_real", "").replace("gdd_real", "").replace("*", "").replace(" ", "")
      if c_v in gdd_decl:
        db_decl.append(c_v)
        gdd_decl.remove(c_v)
      return stmt.replace("::gdd_real", "double").replace("gdd_real", "double")
  if p_var in stmt and p_var not in stmt.split('=')[0] and "double" in stmt.split('=')[0] and "to_double" not in stmt:
    cur_lhs = stmt.split('=')[0] 
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    if if_gdd(cur_rhs, db_decl, fl_decl)==False: #and if_gdd(cur_lhs, db_decl, fl_decl)==False:
      c_v = stmt.split('=')[0].replace("double3", "").replace("double", "").replace("double", "").replace("*", "").replace(" ", "")
      if c_v in db_decl:
        fl_decl.append(c_v)
        db_decl.remove(c_v)
      return stmt.replace("double", "float")
  if p_var in stmt and p_var not in stmt.split('=')[0] and "gdd_real" not in stmt and "dd_add" not in stmt:
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    cur_lhs = stmt.split('=')[0] 
    if if_gdd(cur_rhs, gdd_decl, db_decl)==False and if_gdd(cur_lhs, gdd_decl, db_decl)==True:
      new_rhs = "dd_add(" + cur_rhs+", 0.0)"
      return stmt.replace(cur_rhs, new_rhs)
  #check the types in lhs and rhs, to balance the type
  cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
  cur_lhs = stmt.split('=')[0] 
  if if_gdd(cur_rhs, gdd_decl, db_decl)==False and if_gdd(cur_lhs, gdd_decl, db_decl)==True and "dd_add" not in stmt:
    new_rhs = "dd_add(" + cur_rhs+", 0.0)"
    return stmt.replace(cur_rhs, new_rhs)
  elif if_gdd(cur_rhs, gdd_decl, db_decl)==True and if_gdd(cur_lhs, gdd_decl, db_decl)==False and "to_double" not in stmt:
    mod_rhs = "to_double(" + cur_rhs + ")"
    return stmt.replace(cur_rhs, mod_rhs)
  return stmt

def return_type(p_var, gdd_decl, db_decl, stmt):
  if p_var in stmt and "to_double" in stmt:
    ret_stmt = stmt[stmt.find('return ')+len('return '):stmt.rfind(';')] 
    if if_gdd(ret_stmt, gdd_decl, db_decl) == False:
      return stmt.replace("to_double", "")
  return stmt

def find_cur(cur_f, stmt):
   if cur_f == stmt.split('(')[0].split(' ')[-1] and ";" not in stmt:
     return True
   return False

def if_var(cur_v, rhs):
  #hs_list = ''.join([i for i in rhs if not i.isdigit()]) 
  hs_list = rhs 
  if "[" in hs_list:
    tod = re.findall(r"\[(.*?)\]", hs_list)
    for var in tod:
      hs_list = hs_list.replace("["+var+"]", "")
  hs = hs_list.replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").replace("=", " ").replace("(", " ").replace(")", " ").replace("to_double", " ").replace("double3", " ").replace("double", " ").replace(":", " ").replace("gdd_real3", " ").replace ("gdd_real", " ").replace("dd_real3", " ").replace("dd_real", " ").replace(".", " ").replace(",", " ").replace("&", " ").replace("sqrt", " ").replace("exp", " ").replace("[", " ").replace("]", " ").replace(";", "").replace("\n", "").split(" ")
  hs = filter(None, hs)
  if cur_v in hs:
    return True
  return False

def clean_var(var_list):
  for index, var in enumerate(var_list):
    if "[" in var:
      tod = re.findall(r"\[(.*?)\]", var)
      for sub in tod:
        new = var.replace(sub, "").replace("[", "").replace("]", "")
        var_list[index] = new.replace("&", "")

def get_var(cur_v, rhs, offset):
  #hs_list = ''.join([i for i in rhs if not i.isdigit()]) 
  hs_list = rhs 
  if "[" in hs_list:
    tod = re.findall(r"\[(.*?)\]", hs_list)
    for var in tod:
      hs_list = hs_list.replace("["+var+"]", "")
  hs = hs_list.replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").replace("=", " ").replace("(", " ").replace(")", " ").replace("<", " ").replace(">", " ").replace("to_double", " ").replace("double3", " ").replace("double", " ").replace("::", " ").replace("gdd_real3", " ").replace ("gdd_real", " ").replace("dd_real3", " ").replace("dd_real", " ").replace(".", " ").replace(",", " ").replace("&", " ").replace("sqrt", " ").replace("exp", " ").replace("[", " ").replace("]", " ").replace(";", "").replace("\n", "").split(" ")
  hs = filter(None, hs)
  return hs[hs.index(cur_v)+offset]

def var_in_decl(cur_v, rhs):
  hs_list = rhs 
  if "[" in hs_list:
    tod = re.findall(r"\[(.*?)\]", hs_list)
    for var in tod:
      hs_list = hs_list.replace("["+var+"]", "")
  hs = hs_list.replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").replace("=", " ").replace("(", " ").replace(")", " ").replace("to_double", " ").replace("double3", " ").replace("double", " ").replace(":", " ").replace("gdd_real3", " ").replace ("gdd_real", " ").replace("dd_real3", " ").replace("dd_real", " ").replace(".", " ").replace(",", " ").replace("&", " ").replace("sqrt", " ").replace("exp", " ").replace("[", " ").replace("]", " ").replace(";", "").replace("\n", "").split(" ")
  hs = filter(None, hs)
  if cur_v in hs:
    return True
  return False

def if_var_decl(cur_v, stmt):
  if "=" in stmt and cur_v in stmt.split('=')[0]:
    rhs = stmt.split('=')[0]
    return var_in_decl(cur_v, rhs)
  elif "=" not in stmt and cur_v in stmt:
    rhs = stmt.replace(";", "")
    return var_in_decl(cur_v, rhs)
  return False

def if_var_in_stmt(cur_v, stmt):
  if "=" in stmt and cur_v in stmt.split('=')[0]:
    rhs = stmt.split('=')[0]
    return if_var(cur_v, rhs)
  elif "=" not in stmt and cur_v in stmt:
    rhs = stmt.replace(";", "")
    return if_var(cur_v, rhs)
  return False
 
def get_func_name(stmt):
  #print "func name+++++ " + stmt
  return stmt.split('(')[0].split(' ')[-1]

def get_host_dv(cur_v, func, dv_host):
  host_gdd = []
  host_ndd = []
  for stmt in func:
    if (("cudaMemcpy" in stmt and "HostToDevice" in stmt) or "upload" in stmt or "cudaMemcpyToSymbol" in stmt)  and if_var(cur_v, stmt):
      host_gdd.append(get_var(cur_v, stmt, 1).replace("\n", ""))
      #dv_host.append(get_var(cur_v, stmt, 1))
  for host_d in host_gdd:
    for stmt in func:
      if "qd2gqd" in stmt and if_var(host_d, stmt):
        host_ndd.append(get_var(host_d, stmt, -1).replace("\n", ""))
  for idx, item in enumerate(host_gdd):
    dv_host.append(item)
    if idx < len(host_ndd):
      dv_host.append(host_ndd[idx])
def change_type(cur_v, stmt):
  if "=" in stmt and cur_v in stmt.split('=')[0]:
    if if_var(cur_v, stmt.split('=')[0])==False:
      return stmt
    if "::gdd_real" in stmt:
      stmt = stmt.replace("::gdd_real", "double").replace("gdd_real", "double")
      return stmt
    elif "gdd_real" in stmt:
      stmt = stmt.replace("gdd_real", "double")
      return stmt
    if "::dd_real" in stmt:
      stmt = stmt.replace("::dd_real", "double").replace("dd_real", "double")
      return stmt
    elif "dd_real" in stmt:
      stmt = stmt.replace("dd_real", "double")
      return stmt
    
    if "double" in stmt and "(double)" not in stmt.split('=')[0] and "(double )" not in stmt.split('=')[0]:
      stmt = stmt.replace("double", "float")
      return stmt
  elif "=" not in stmt:
    if if_var(cur_v, stmt.replace(";", ""))==False:
      return stmt
    if "::gdd_real" in stmt:
      stmt = stmt.replace("::gdd_real", "double")
      return stmt
    elif "gdd_real" in stmt:
      stmt = stmt.replace("gdd_real", "double")
      return stmt
    if "::dd_real" in stmt:
      stmt = stmt.replace("::dd_real", "double")
      return stmt
    elif "dd_real" in stmt:
      stmt = stmt.replace("dd_real", "double")
      return stmt
    elif "double" in stmt:
      stmt = stmt.replace("double", "float")
      return stmt
  return stmt

def target_node(r_f, p_f, stmt):
  f_name = stmt.split('(')[0].split(' ')[-1] 
  if "dd_add" in f_name:
    return True
  if "," in stmt[stmt.find('(')+len('('):stmt.rfind(')')] :
    return False
  if f_name in r_f:
    return False
  if f_name in p_f:
    return False
  return True

def print_func(func, gdd_decl, db_decl):
  print "func: \n"
  print "\tgdd: \n"
  for gdd in gdd_decl:
    print "\t\t" + gdd 
  print "\tdb: \n"
  for db in db_decl:
    print "\t\t" + db 
  for stmt in func:
    print stmt

def write_func(func):
  for stmt in func:
    print stmt

def func_update(func, gdd_decl, db_decl, fl_decl, ret_f, param_f, cur_f, cur_v, is_single):
  #get all variables in this function with gdd_real and double type
  for stmt in func:
    #process variable decl in parameter
    if "(" in stmt and ";" not in stmt:
      param_list = stmt[stmt.find('(')+len('('):stmt.rfind(')')].split(',')
      clean_var(param_list)
      for param in param_list:
        if "gdd_real" in param:
          gdd_decl.append(param.replace("gdd_real3", "").replace("gdd_real", "").replace("&", "").replace("*", "").replace("::", "").replace(" ", ""))
        elif "double" in param:
          db_decl.append(param.replace("double3", "").replace("db", "").replace("&", "").replace("*", "").replace(" ", ""))
        elif "float" in param:
          fl_decl.append(param.replace("flaot3", "").replace("&", "").replace("*", "").replace(" ", ""))
    elif "gdd_real" in stmt and ";" in stmt:
      if "=" in stmt:
        gdd_decl.append(stmt.replace("gdd_real3", "").replace("gdd_real", "").replace(" ", "").replace("*", "").replace("::", "").split('=')[0]);
      elif "[" in stmt:
        gdd_decl.append(stmt.replace("__shared__", "").replace("gdd_real3", "").replace("gdd_real", "").replace(" ", "").replace("*", "").replace("::", "").split('[')[0]);
      else:
        gdd_decl.append(stmt.replace("gdd_real3", "").replace("gdd_real", "").replace(" ", "").replace("*", "").replace("::", "").split(';')[0]);
    elif ";" in stmt and "=" in stmt and "double" in stmt:
      if "double" in stmt.split('=')[0]:
        db_decl.append(stmt.replace("double3", "").replace("double", "").replace(" ", "").replace("*", "").split('=')[0]);
    elif ";" in stmt and "double" in stmt and "=" not in stmt and "return" not in stmt:
      if "[" in stmt:
        db_decl.append(stmt.replace("__shared__", "").replace("double3", "").replace("double", "").replace(" ", "").replace("*", "").split('[')[0]);
      else:
        db_decl.append(stmt.replace("double3", "").replace("double", "").replace(" ", "").replace("*", "").split(';')[0]);
    elif ";" in stmt and "=" in stmt and "float" in stmt:
      if "float" in stmt.split('=')[0]:
        fl_decl.append(stmt.replace("float3", "").replace("float", "").replace(" ", "").replace("*", "").split('=')[0]);
    elif ";" in stmt and "float" in stmt and "=" not in stmt and "return" not in stmt:
      if "[" in stmt:
        fl_decl.append(stmt.replace("__shared__", "").replace("float3", "").replace("float", "").replace(" ", "").replace("*", "").split('[')[0]);
      else:
        db_decl.append(stmt.replace("float3", "").replace("float", "").replace(" ", "").replace("*", "").split(';')[0]);
  is_mod = False
  is_param = False
  is_target = False
  mod_param = None
  param_group = []
  for index, stmt in enumerate(func): 
   # check if this function needs to fix return type
   if find_ret(ret_f, stmt):
     is_mod = True
     func[index] = change_ret(stmt)
   if is_mod == True:
    new_stmt = find_return(gdd_decl, stmt)
    if new_stmt != stmt:
      func[index] = new_stmt
      is_mod = False
   # check if this function needs to fix function parameter
   p_group = find_param(param_f, stmt)
   if p_group != -1:
     is_param = True
     #del param_f[stmt.split('(')[0].split(' ')[-1]]
     for p_id in p_group: 
       mod_param = get_param(p_id, stmt)
       stmt = change_param(p_id, stmt, is_single)
       param_group.append(mod_param)
       if mod_param in gdd_decl:
         db_decl.append(mod_param)
         gdd_decl.remove(mod_param)
       elif mod_param in db_decl:
         fl_decl.append(mod_param)
         db_decl.remove(mod_param)
     func[index] = stmt
   if is_param:
     for mod_param in param_group: 
       #update list of gdd_real and double after change parameter
       if mod_param in stmt and "=" in stmt:
         new_stmt = balance_type(mod_param, gdd_decl, db_decl, fl_decl, stmt)
         if new_stmt != stmt:
           func[index] = new_stmt
       if mod_param in stmt and "return" in stmt:
         new_stmt = return_type(mod_param, gdd_decl, db_decl, stmt)
         if new_stmt != stmt:
           func[index] = new_stmt
       if "=" in stmt:
         new_stmt = balance_type(mod_param, gdd_decl, db_decl, fl_decl, stmt)
         if new_stmt != stmt:
           func[index] = new_stmt
   # check if this function needs to fix function parameter
   if find_cur(cur_f, stmt):
     is_target = True
   if is_target == True and cur_v in stmt and ("gdd_real" in stmt or ("double" in stmt and is_single)):
     new_stmt = change_type(cur_v, stmt)
     if cur_v in gdd_decl:
       db_decl.append(cur_v)
       gdd_decl.remove(cur_v)
     elif cur_v in db_decl:
       fl_decl.append(cur_v)
       db_decl.remove(cur_v)
     if "=" in new_stmt and ");" not in new_stmt:
       #if cur_v == "t1":
       #  print "var--- " + new_stmt;
       new_stmt = balance_type(cur_v, gdd_decl, db_decl, fl_decl, new_stmt)
     if "=" in new_stmt and ")" in new_stmt:
      if target_node(ret_f, param_f, stmt):
        new_stmt = balance_type(cur_v, gdd_decl, db_decl, fl_decl, new_stmt)
     if new_stmt != stmt:
       func[index] = new_stmt
     continue
   if is_target == True and "=" in stmt and cur_v in stmt and ")" in stmt:
     if target_node(ret_f, param_f, stmt):
       new_stmt = balance_type(cur_v, gdd_decl, db_decl, fl_decl, stmt)
       if new_stmt != stmt:
         func[index] = new_stmt
   if is_target == True and cur_v in stmt and "=" in stmt and ");" not in stmt:
     new_stmt = balance_type(cur_v, gdd_decl, db_decl, fl_decl, stmt)
     if new_stmt != stmt:
       func[index] = new_stmt

def update_use_of_var(file_list, target_func, target_var, ret_func, param_func):
  inside_func = False
  f_cnt = 0
  #variables defined and tuned in current function
  for stmt in file_list:
    if not stmt.strip() or "//" in stmt:
      continue
    if target_func == stmt.split('(')[0].split(' ')[-1]:
      #check function boundary
      if ";" in stmt:
        continue
      inside_func = True
      if "{" in stmt:
        f_cnt += 1
    elif inside_func and target_var in stmt:
      #check function boundary
      if "{" in stmt:
        f_cnt += 1
      elif "}" in stmt:
        f_cnt -= 1
      #check the functions that related to this variable
      if ");" in stmt and is_func(stmt)==True:
        #check if the variable is the left value
        params = stmt[stmt.find('(')+len('('):stmt.rfind(')')].replace("&", "").split(',')
        func_name = stmt.split('(')[0].split(' ')[-1]
        clean_var(params)
        #clean the parameters that has [], &
        if "=" in stmt and target_var in stmt.split('=')[0] and is_math(func_name)==False and "dd_add" not in stmt:
          ret_func.append(func_name)
        elif target_var in params and "qd2gqd" not in stmt and func_name not in param_func: 
          seen = set()
          for idx, item in enumerate(params):
            if item not in seen:
              seen.add(item)
              if func_name not in param_func and target_var==item:
                param_func[func_name] = [idx]
            else:
              if target_var==item:
                param_func[func_name].append(idx)
          #if target_var == "rhs":
          #  print "param: " +  func_name + ", index: " + str(param_func[func_name])
          #  print params
    elif inside_func:
      #check function boundary
      if "{" in stmt:
        f_cnt += 1
      elif "}" in stmt:
        f_cnt -= 1
      if f_cnt == 0:
        break

def update_precision(ret_func, param_func, dv_g, db_g, fl_g, target_func, target_var, is_single):
  with closing(fileinput.FileInput(sys.argv[2], inplace=True, backup='.bak')) as stmt_list:
    func_call = []
    gdd_def = []
    db_def = []
    fl_def = []
    inside_func = False
    f_cnt = 0
    for stmt in stmt_list:
      #skip the empty line
      if not stmt.strip():
        continue
      if if_target(ret_func, param_func, target_func, stmt):
        #clear the func_call 
        del func_call[:]
        del gdd_def[:]
        del db_def[:]
        inside_func = True
        if "{" in stmt:
          f_cnt += 1
        #add stmt into target function call list
        func_call.append(stmt)
      elif inside_func:
        if "{" in stmt:
          f_cnt += 1
        elif "}" in stmt:
          f_cnt -= 1
        #add stmt into target function call list
        func_call.append(stmt)
        #function load complete, modification begin
        if f_cnt == 0:
          #load global defined device function into gdd_def list 
          for dv_var in dv_g:
            gdd_def.append(dv_var)
          for db_var in db_g:
            db_def.append(db_var)
          for fl_var in fl_g:
            fl_def.append(fl_var)
          func_update(func_call, gdd_def, db_def, fl_def, ret_func, param_func, target_func, target_var, is_single)
          #print_func(func_call, gdd_def, db_def)
          write_func(func_call)
          inside_func = False
      #other stmt, just store into the original file
      else:
        print stmt

def host_balance_type(p_var, gdd_decl, db_decl, fl_decl, stmt):

  if p_var in stmt.split('=')[0] and "to_double" not in stmt:
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    cur_lhs = stmt.split('=')[0] 
    if "float" in cur_lhs and "(float)" not in stmt and if_gdd(cur_rhs, db_decl, fl_decl):
      mod_rhs = "(float)" + cur_rhs
      stmt = stmt.replace(cur_rhs, mod_rhs)
      return stmt
    if if_gdd(cur_rhs, gdd_decl, db_decl) and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
      mod_rhs = "to_double(" + cur_rhs + ")"
      stmt = stmt.replace(cur_rhs, mod_rhs)
      return stmt
  if p_var in stmt and p_var not in stmt.split('=')[0] and "to_double" in stmt:
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    if if_gdd(cur_rhs, gdd_decl, db_decl)== False:
      return stmt.replace("to_double", "")
  if p_var in stmt and p_var not in stmt.split('=')[0] and "dd_real" in stmt:
    cur_lhs = stmt.split('=')[0] 
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    #if if_gdd(cur_rhs, gdd_decl, db_decl)==False and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
    if if_gdd(cur_rhs, gdd_decl, db_decl)==False:# and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
      c_v = stmt.split('=')[0].replace("::dd_real3", "").replace("::dd_real", "").replace("dd_real", "").replace("*", "").replace(" ", "")
      if c_v in gdd_decl:
        db_decl.append(c_v)
        gdd_decl.remove(c_v)
      return stmt.replace("::dd_real", "double").replace("dd_real", "double")
  if p_var in stmt and p_var not in stmt.split('=')[0] and "dd_real" in stmt and "gdd_real" not in stmt:
    cur_lhs = stmt.split('=')[0] 
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    if if_gdd(cur_rhs, gdd_decl, db_decl)==False and if_gdd(cur_lhs, gdd_decl, db_decl)==False:
      c_v = stmt.split('=')[0].replace("::dd_real3", "").replace("::dd_real", "").replace("dd_real", "").replace("*", "").replace(" ", "")
      if c_v in gdd_decl:
        db_decl.append(c_v)
        gdd_decl.remove(c_v)
      return stmt.replace("::dd_real", "double").replace("dd_real", "double")
  if p_var in stmt and p_var not in stmt.split('=')[0] and "double" in stmt:
    cur_lhs = stmt.split('=')[0] 
    cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
    if if_gdd(cur_rhs, db_decl, fl_decl)==False: #and if_gdd(cur_lhs, db_decl, fl_decl)==False:
    #if if_gdd(cur_rhs, db_decl, fl_decl)==False and if_gdd(cur_lhs, db_decl, fl_decl)==False:
      c_v = stmt.split('=')[0].replace("double3", "").replace("double", "").replace("double", "").replace("*", "").replace(" ", "")
      if c_v in gdd_decl:
        fl_decl.append(c_v)
        db_decl.remove(c_v)
      return stmt.replace("double", "float")
  #check the types in lhs and rhs, to balance the type
  cur_rhs = stmt[stmt.find('=')+len('='):stmt.rfind(';')]
  cur_lhs = stmt.split('=')[0] 
  if if_gdd(cur_rhs, gdd_decl, db_decl)==True and if_gdd(cur_lhs, gdd_decl, db_decl)==False and "to_double" not in stmt:
    mod_rhs = "to_double(" + cur_rhs + ")"
    return stmt.replace(cur_rhs, mod_rhs)
  return stmt

# modify and test balance_type
def host_func_update(func, dd_decl, db_decl, fl_decl, ret_f, param_f, cur_f, hd_var, is_fl):
  #get all variables in this function with gdd_real and double type
  cur_v = ''

  if len(hd_var) > 1:
    cur_v = hd_var[1]
    h_gdd = hd_var[0]
  elif len(hd_var) ==1:
    cur_v = hd_var[0]

  for stmt in func:
    #process variable decl in parameter
    if "(" in stmt and ";" not in stmt:
      param_list = stmt[stmt.find('(')+len('('):stmt.rfind(')')].split(',')
      clean_var(param_list)
      for param in param_list:
        if "dd_real" in param:
          dd_decl.append(param.replace("dd_real3", "").replace("dd_real", "").replace("&", "").replace("*", "").replace("::", "").replace(" ", ""))
        elif "double" in param:
          db_decl.append(param.replace("double3", "").replace("db", "").replace("&", "").replace("*", "").replace(" ", ""))
    elif "dd_real" in stmt and ";" in stmt:
      if "=" in stmt:
        dd_decl.append(stmt.replace("dd_real3", "").replace("dd_real", "").replace(" ", "").replace("*", "").replace("::", "").split('=')[0]);
      elif "[" in stmt:
        dd_decl.append(stmt.replace("__shared__", "").replace("dd_real3", "").replace("dd_real", "").replace(" ", "").replace("*", "").replace("::", "").split('[')[0]);
      else:
        dd_decl.append(stmt.replace("dd_real3", "").replace("dd_real", "").replace(" ", "").replace("*", "").replace("::", "").split(';')[0]);
    elif ";" in stmt and "=" in stmt and "double" in stmt:
      if "double" in stmt.split('=')[0]:
        db_decl.append(stmt.replace("double3", "").replace("double", "").replace(" ", "").replace("*", "").split('=')[0]);
    elif ";" in stmt and "double" in stmt and "=" not in stmt and "return" not in stmt:
      if "[" in stmt:
        db_decl.append(stmt.replace("__shared__", "").replace("double3", "").replace("double", "").replace(" ", "").replace("*", "").split('[')[0]);
      else:
        db_decl.append(stmt.replace("double3", "").replace("double", "").replace(" ", "").replace("*", "").split(';')[0]);
    elif ";" in stmt and "=" in stmt and "float" in stmt:
      if "float" in stmt.split('=')[0]:
        fl_decl.append(stmt.replace("float3", "").replace("float", "").replace(" ", "").replace("*", "").split('=')[0]);
    elif ";" in stmt and "float" in stmt and "=" not in stmt and "return" not in stmt:
      if "[" in stmt:
        fl_decl.append(stmt.replace("__shared__", "").replace("float3", "").replace("float", "").replace(" ", "").replace("*", "").split('[')[0]);
      else:
        fl_decl.append(stmt.replace("float3", "").replace("float", "").replace(" ", "").replace("*", "").split(';')[0]);
  is_mod = False
  is_param = False
  is_target = False
  mod_param = None
  param_group = []
  stmt_remove = []
  for index, stmt in enumerate(func): 
    # clear the gdd variables when tuning high-precision down to double/float
    if len(hd_var) > 1:
      if if_var(h_gdd, stmt)==True and "cuda" not in stmt and "upload" not in stmt:
        stmt_remove.append(stmt)
        continue
      if if_var(h_gdd, stmt) and ("qd2gqd" in stmt or "gqd2qd" in stmt):
        stmt_remove.append(stmt)
        continue
      if if_var(h_gdd, stmt)==True and ("cuda" in stmt or "upload" in stmt):
        if "double" in stmt and is_fl:
          func[index] = stmt.replace("double", "float")
        else:
          func[index] = stmt.replace(h_gdd, cur_v).replace("gdd_real", "double")
    # check if this function needs to fix return type
    if find_ret(ret_f, stmt):
      is_mod = True
      func[index] = change_ret(stmt)
    if is_mod == True:
     new_stmt = find_return(dd_decl, stmt)
     if new_stmt != stmt:
       func[index] = new_stmt
       is_mod = False
    # check if this function needs to fix function parameter
    p_group = find_param(param_f, stmt)
    if p_group != -1:
      is_param = True
      for p_id in p_group: 
       mod_param = get_param(p_id, stmt)
       stmt = change_param(p_id, stmt, False)
       param_group.append(mod_param)
       if mod_param in dd_decl:
         db_decl.append(mod_param)
         dd_decl.remove(mod_param)
       elif mod_param in db_decl:
         fl_decl.append(mod_param)
         db_decl.remove(mod_param)
      func[index] = stmt
    if is_param:
      for mod_param in param_group: 
        #update list of gdd_real and double after change parameter
        if mod_param in stmt and "=" in stmt:
          new_stmt = host_balance_type(mod_param, dd_decl, db_decl, fl_decl, stmt)
          if new_stmt != stmt:
            func[index] = new_stmt
        if mod_param in stmt and "return" in stmt:
          new_stmt = return_type(mod_param, dd_decl, db_decl, stmt)
          if new_stmt != stmt:
            func[index] = new_stmt
        if "=" in stmt:
          new_stmt = host_balance_type(mod_param, dd_decl, db_decl, fl_decl, stmt)
          if new_stmt != stmt:
            func[index] = new_stmt
    # check if this function needs to fix function parameter
    if find_cur(cur_f, stmt):
      is_target = True

    if is_target == True and cur_v in stmt and ("dd_real" in stmt or ("double" in stmt and is_fl)):
      new_stmt = change_type(cur_v, stmt)
      if new_stmt != stmt:
        if cur_v in dd_decl:
          db_decl.append(cur_v)
          dd_decl.remove(cur_v)
        elif cur_v in db_decl:
          fl_decl.append(cur_v)
          db_decl.remove(cur_v)
      if "=" in new_stmt and ");" not in new_stmt:
        new_stmt = host_balance_type(cur_v, dd_decl, db_decl, fl_decl, new_stmt)
      if "=" in new_stmt and ")" in new_stmt:
       if target_node(ret_f, param_f, stmt):
         new_stmt = host_balance_type(cur_v, dd_decl, db_decl, fl_decl, new_stmt)
      if new_stmt != stmt:
        func[index] = new_stmt
      continue
    if is_target == True and "=" in stmt and cur_v in stmt and ")" in stmt:
      if target_node(ret_f, param_f, stmt):
        new_stmt = host_balance_type(cur_v, dd_decl, db_decl, fl_decl, stmt)
        if new_stmt != stmt:
          func[index] = new_stmt
    if is_target == True and cur_v in stmt and "=" in stmt and ");" not in stmt:
      new_stmt = host_balance_type(cur_v, dd_decl, db_decl, fl_decl, stmt)
      if new_stmt != stmt:
        func[index] = new_stmt
  for stmt in stmt_remove:
    func.remove(stmt)

def update_host_precision(ret_func, param_func, dv_g, db_g, fl_g, d_def, target_func, h_glb, is_fl):
  with closing(fileinput.FileInput(sys.argv[2], inplace=True, backup='.bak')) as stmt_list:
  #with open(sys.argv[2], "r") as stmt_list:
    func_call = []
    dd_def = []
    db_def = []
    fl_def = []
    inside_func = False
    f_cnt = 0
    for stmt in stmt_list:
      #skip the empty line
      if not stmt.strip():
        continue
      if d_def != None and d_def in stmt:
        if is_fl and "double" in stmt:
          print(stmt.replace("double", "float"))
        else:
          print(stmt.replace("::gdd_real", "double").replace("gdd_real", "double"))
      elif if_target(ret_func, param_func, target_func, stmt):
        #clear the func_call 
        del func_call[:]
        del dd_def[:]
        del db_def[:]
        inside_func = True
        if "{" in stmt:
          f_cnt += 1
        #add stmt into target function call list
        func_call.append(stmt)
      elif inside_func:
        if "{" in stmt:
          f_cnt += 1
        elif "}" in stmt:
          f_cnt -= 1
        #add stmt into target function call list
        func_call.append(stmt)
        #function load complete, modification begin
        if f_cnt == 0:
          #load global defined device function into dd_def list 
          for dv_var in dv_g:
            dd_def.append(dv_var)
          for db_var in db_g:
            db_def.append(db_var)
          for fl_var in fl_g:
            fl_def.append(fl_var)
          host_func_update(func_call, dd_def, db_def, fl_def, ret_func, param_func, target_func, h_glb, is_fl)
          #print_func(func_call, dd_def, db_def)
          write_func(func_call)
          inside_func = False
      #other stmt, just store into the original file
      else:
        print stmt
"""""""""""""""""""""""""""
" variable definition section
"""""""""""""""""""""""""""
      
dev_vars = []
host_vars = []
db_vars = []
gdd_vars = []
fl_vars = []
sd_vars = []

dv_glb = []
dev_host = []
host_host = []

host_func = []
dev_func = []
insert_list = []
host_new = {}
new_size = {}
new_ptr = {}
call_cnt = {}

comp_func = []
call_func = []

var_combo = {}

#comp_vars = {}
comp_vars = OrderedDict()

visit = 0

"""""""""""""""""""""""""""
" main function section
"""""""""""""""""""""""""""
#run the translator for the original program
subprocess.call(['./Translator ' + str(sys.argv[1])], shell=True)

ben_dir = sys.argv[3]

#read device variables list from file
with open(ben_dir+"device_vars.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    dev_vars.append(line)
    #print line

with open(ben_dir+"host_vars.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    host_vars.append(line)
    #print line

with open(ben_dir+"special_vars.txt", "r") as var_f:
  db = False
  gdd = False
  dev_glb = False
  dev = False
  host = False
  fl = False
  sd = False
  for line in var_f:
    if "sd:" in line:
      sd = True
      fl = False
      db = False
      dev_glb = False
      gdd = False
      dev = False
      host = False
      continue
    if "fl:" in line:
      fl = True
      sd = False
      db = False
      dev_glb = False
      gdd = False
      dev = False
      host = False
      continue
    if "db:" in line:
      db = True
      sd = False
      fl = False
      dev_glb = False
      gdd = False
      dev = False
      host = False
      continue
    if "gdd_real:" in line:
      fl = False
      sd = False
      db = False
      dev_glb = False
      gdd = True
      dev = False
      host = False
      continue
    if "device_global:" in line:
      fl = False
      gdd = False
      db = False
      dev_glb = True
      dev = False
      host = False
      continue
    if "device_host:" in line:
      fl = False
      sd = False
      gdd = False
      db = False
      dev_glb = False
      dev = True
      host = False
      continue
    if "host_host:" in line:
      fl = False
      sd = False
      gdd = False
      db = False
      dev_glb = False
      dev = False
      host = True
      continue
    if sd==True:
      line = line.replace("\n", "");
      sd_vars.append(line)
      print line
    if fl==True:
      line = line.replace("\n", "");
      fl_vars.append(line)
      print line
    if db==True:
      line = line.replace("\n", "");
      db_vars.append(line)
      print line
    elif gdd==True:
      line = line.replace("\n", "");
      gdd_vars.append(line)
      print line
    elif dev_glb==True:
      line = line.replace("\n", "");
      dv_glb.append(line)
    elif dev==True:
      line = line.replace("\n", "");
      dev_host.append(line)
    elif host==True:
      line = line.replace("\n", "");
      host_host.append(line)

with open(ben_dir+"host_func.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    host_func.append(line)
    #print line
with open(ben_dir+"func_list.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    dev_func.append(line)

with open(ben_dir+"func_insert.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    insert_list.append(line)
    #print line

def if_call(line):
  counter = 0
  with open(sys.argv[1], "r") as var_f:
    for stmt in var_f:
      if line in stmt:
        counter += 1
  if counter > 2:
    return False
  return True

with open(ben_dir+"compute_func.txt", "r") as var_f:
  #print "compute func: \n"
  var_read = False
  cur_func = None
  for line in var_f:
    if ":" in line:
      line = line.replace("\n", "").replace(":", "")
      #if if_call(line):
      comp_func.append(line)
      cur_func = line
      var_read = True
      #else:
      #  call_func.append(line)
    elif var_read:
      if not line.strip():
        continue
      input_var = line.replace("\n", "")
      input_split =  input_var.split(' ')
      for item in input_split:
        if not line.strip():
          continue
        if item.isdigit():
          var_combo[input_var] = item
        else:
          input_var = item
          if cur_func not in comp_vars:
            comp_vars[cur_func] = [input_var]
          else:
            comp_vars[cur_func].append(input_var)
print "compute_func\n"
for func in comp_vars:
  print func
  for var in comp_vars[func]:
    print "\t"+var
print str(var_combo)

with open(ben_dir+"call_func.txt", "r") as var_f:
  #print "call func: \n"
  for line in var_f:
    if ":" in line:
      line = line.replace("\n", "").replace(":", "")
      call_func.append(line)
      #print line
#maximize the input program
counter = 0
with closing(fileinput.FileInput(sys.argv[2], inplace=True, backup='.bak')) as stmt_list:
  f_cnt = 0
  for stmt_line in stmt_list:
    if counter == 0:
      for ins_line in insert_list:
        print(ins_line)
    if "+=" in stmt_line:
      if (is_Ftype(stmt_line.split('+=')[0])):
        l_val = stmt_line.split('+=')[0] 
        stmt_line = stmt_line.replace('+=', ("="+l_val+" + ")) 
    elif "-=" in stmt_line:
      if (is_Ftype(stmt_line.split('-=')[0])):
        l_val = stmt_line.split('-=')[0] 
        stmt_line = stmt_line.replace('-=', ("="+l_val+" - ")) 
    elif "*=" in stmt_line:
      if (is_Ftype(stmt_line.split('*=')[0])):
        l_val = stmt_line.split('*=')[0] 
        stmt_line = stmt_line.replace('*=', ("="+l_val+" * ")) 
    elif "/=" in stmt_line:
      if (is_Ftype(stmt_line.split('/=')[0])):
        l_val = stmt_line.split('/=')[0] 
        stmt_line = stmt_line.replace('/=', ("="+l_val+" / ")) 
    #if "//" in stmt_line:
    #  counter += 1
    #  continue
    #if "main" in stmt_line or "set_constants" in stmt_line:
    #  visit = 1
      #if "{" in stmt_line:
      #  f_cnt += 1
    #  print stmt_line
    #elif visit == 1:
      #if "{" in stmt_line:
      #  f_cnt += 1
      #elif "}" in stmt_line:
      #  f_cnt -= 1
    if is_global(dv_glb, stmt_line) and "cuda" not in stmt_line:
      if "double" in stmt_line and "=" not in stmt_line and "(double)" not in stmt_line:
        if "constant" not in stmt_line:
          print(stmt_line.replace('double', '__constant__ gdd_real').replace("::", ""))
          continue
        else:
          print(stmt_line.replace('double', 'gdd_real'))
          continue
      else:
        print stmt_line
        continue
    if is_gdd(host_vars, stmt_line): 
      gdd_assign(host_vars, stmt_line)
    else:
      ret = max_pattern(dev_vars, host_vars, db_vars, fl_vars, sd_vars, host_new, new_size, new_ptr, stmt_line) 
      if ret == 1:
        if "gdd_real" not in stmt_line:
          print(stmt_line.replace('dd_real', 'gdd_real').replace('double', 'gdd_real'))
        else:
          print(stmt_line.replace('double', 'gdd_real'))
      elif ret == 11:
        counter += 1
        continue;
      elif ret == 2:
        print(stmt_line.replace('double', 'dd_real').replace('gdd_real', 'dd_real'))
        #gd_stmt = stmt_line.replace();
      elif ret == 3:
        print(stmt_line.replace('gdd_real', 'double').replace("dd_real", 'double').replace('::', ''))
      elif ret == 4:
        print(stmt_line.replace('gdd_real', 'float').replace("dd_real", 'float').replace('::', ''))
      #if f_cnt == 0:
      #  visit = 0
      elif is_host(host_func, stmt_line):
        print(stmt_line.replace('gdd_real', 'dd_real').replace('double', 'dd_real'))
      elif is_dev(dev_func, stmt_line):
        if "gdd_real" not in stmt_line:
          print(stmt_line.replace('double', 'gdd_real').replace('dd_real', 'gdd_real'))
        else:
          print stmt_line
      else:
        print stmt_line
    counter += 1
#print "-----call: "
#for func in call_func:
#  print func
# collect call function times and duplicate call function 
new_call = []
cur_call = []
call_belong = {}
for func in call_func:
  del cur_call[:]
  func = func.replace("\n", "")
  recur_func(func, cur_call)
  cur_call.remove(func)
  if len(cur_call) != 0:
    call_belong[func] = cur_call
    for f in cur_call:
      new_call.append(f)
#print "-----new call: "
for func in new_call:
  #print func
  call_func.append(func)
  #check the call times for all the call function

for func in call_func:
  func = func.replace("\n", "")
  call_cnt[func] = 0
  with open(sys.argv[2], "r") as stmt_list:
    for stmt_line in stmt_list:
      if func == stmt_line.split('(')[0].split(' ')[-1] and ";" in stmt_line:
        call_cnt[func] += 1
 #  add function call times for inner function
for func in call_belong:
  for f in call_belong[func]:
    call_cnt[f] += call_cnt[func]
print "-----call: "
for func in call_cnt:
 print func + " " +str(call_cnt[func])

# duplicate function call
#with open(sys.argv[2], "r") as stmt_list:
with closing(fileinput.FileInput(sys.argv[2], inplace=True, backup='.bak')) as stmt_list:
  f_begin = False
  f_cnt = 0
  cur_func = None
  func_body = []
  call_id = {}
  for cf in call_func:
    cf = cf.replace("\n", "")
    call_id[cf] = 0
  for stmt_line in stmt_list:
    #stmt_line = stmt_line.strip()
    if not stmt_line.strip():
      print stmt_line
      continue
    if f_begin == False:
      if ";" in stmt_line:
        print(find_call(call_func, call_id, stmt_line))
        continue
      if find_func(call_func, stmt_line):
        f_begin = True
        cur_func = stmt_line
        func_body.append(stmt_line)
        if "{" in stmt_line:
          f_cnt += 1
      print stmt_line
    else:
     func_body.append(stmt_line)
     if "{" in stmt_line:
       f_cnt += 1
     elif "}" in stmt_line:
       f_cnt -= 1
     print stmt_line
     if f_cnt == 0:
       f_begin = False
       #duplicate function
       dup_func(cur_func, func_body, call_cnt)
       del func_body[:]

print_insert = []
cmp_insert = []
cmp_v = []
cmp_x = []
cmp_y = []
db_cmp_insert = []
fl_cmp_insert = []
run_program = []
with open(ben_dir+"print_result.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    print_insert.append(line)
with open(ben_dir+"compare_result.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    cmp_insert.append(line)
#with open(ben_dir+"compare_v.txt", "r") as var_f:
#  for line in var_f:
#    line = line.replace("\n", "");
#    cmp_v.append(line)
#with open(ben_dir+"compare_x.txt", "r") as var_f:
#  for line in var_f:
#    line = line.replace("\n", "");
#    cmp_x.append(line)
#with open(ben_dir+"compare_y.txt", "r") as var_f:
#  for line in var_f:
#    line = line.replace("\n", "");
#    cmp_y.append(line)
with open(ben_dir+"db_compare_result.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    db_cmp_insert.append(line)
with open(ben_dir+"fl_compare_result.txt", "r") as var_f:
  for line in var_f:
    line = line.replace("\n", "");
    fl_cmp_insert.append(line)

with open(sys.argv[2], "r") as var_f:
  del run_program[:]
  for line in var_f:
    line = line.replace("\n", "");
    run_program.append(line)
subprocess.call(['rm -f ' + "run_program.run"], shell=True)
#subprocess.call(['rm -f ' + "run_program.run"], shell=True)
with open("run_program.run", "a") as wline:
  for stmt in run_program:
    if not stmt.strip():
      continue
    #skip the empty line
    if "gettimeofday(&end_t" in stmt:
      wline.write(stmt+"\n")
      for line in print_insert:
        wline.write(line+"\n")
    else:
      wline.write(stmt+"\n")
##run the translator for the original program
subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/cfd/euler3d_dd.cu; rm -f density_ref.txt'], shell=True)
subprocess.call(['cd ' + ' ./cuda_bench/cfd; make clean;make; ./euler3d_dd ../data/cfd/fvcorr.domn.193K'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/cfd; make clean;make; ./euler3d_dd ../data/cfd/fvcorr.domn.097K'], shell=True)
#subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/sp/sp_dd.cu; rm -f u_ref.txt'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/sp; make clean;make; ./sp_dd'], shell=True)
#subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/gaussian/gaussian_dd.cu; rm -f fv_ref.txt;'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/gaussian; make clean;make; ./gaussian_dd -s 1024'], shell=True)
#subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/hotspot/hotspot_dd.cu; rm -f out_ref.txt;'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/hotspot; make clean;make; ./hotspot_dd 1024 2 2 ../data/hotspot/temp_1024 ../data/hotspot/power_1024'], shell=True)
#subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/lud/lud_dd.cu; rm -f m_ref.txt;'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/lud; make clean;make; ./lud_dd -s 2048'], shell=True)
#subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/lavaMD/lava_dd.cu; rm -f fv_ref.txt;'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/lavaMD; make clean;make; ./lava_dd -boxes1d 24'], shell=True)
#exit()
print "\ncalling func:\n"
#go through the variable list in function to tune down precision
is_run = False
is_single = False
skip_factor = 0
skip_cnt = 0
is_skip_for_var_combo = False
tune_round = 2
global_var = []
#initiazlize global variable list
for var in dv_glb:
  global_var.append(var)

for x in range(0, tune_round):
  for func in comp_vars:
    func = func.replace("\n", "")
    is_glb = False
    for cur_var in comp_vars[func]:
      is_glb = False
      param_func = {}
      ret_func = []
      param_f = []
      hd_gld = []
      dv_def = None
      dv_func = []
      db_glb = []
      fl_glb = []
      cur_var = cur_var.replace("\n", "")
      with open(sys.argv[2], "r") as stmt_list:
        inside_func = False
        f_cnt = 0
        #global device variable tuned in current function
          #ret_func used to store function that handles cudaMemcpy
          #param_func used to store function that use device global variables 
          #dv_gld stores the decl instruction
          #hd_gld stores the host variables that copy the value
        if cur_var in dv_glb or cur_var in dev_host or cur_var in host_host or cur_var in global_var:
          is_glb = True
          #if cur_var == "tx1":
          #print "cur---"
          #print dv_glb
          #print host_host
          for stmt in stmt_list:
            if not stmt.strip():
              continue
            # decl instruction 
            if cur_var in global_var:
              if "gdd_real" in stmt and "cuda" not in stmt and "dealloc" not in stmt  and "upload" not in stmt and "alloc" not in stmt and if_var_in_stmt(cur_var, stmt)==True:
                dv_def = stmt
              elif "double" in stmt and is_single and "cuda" not in stmt and "dealloc" not in stmt  and "upload" not in stmt and "alloc" not in stmt and if_var_in_stmt(cur_var, stmt)==True: 
                dv_def = stmt
            # param function 
            if "(" in stmt and ")" in stmt and ";" not in stmt and "if" not in stmt and "else" not in stmt and "while" not in stmt and "switch" not in stmt and "for" not in stmt and "#define" not in stmt:
              #print "func==== " + stmt
              inside_func = True
              del dv_func[:]
              #clear the func_call 
              if "{" in stmt:
                f_cnt += 1
              #add stmt into target function call list
              dv_func.append(stmt)
            elif inside_func:
              if "{" in stmt:
                f_cnt += 1
              elif "}" in stmt:
                f_cnt -= 1
              #add stmt into target function call list
              dv_func.append(stmt)
              #function load complete, modification begin
              if f_cnt == 0:
                inside_func = False
                #print "end func==== " + dv_func[0]
                if "main" in dv_func[0]:
                #if "main" in dv_func[0] and is_single==False:
                  # check the related varibales in main function for the global variable type change
                  get_host_dv(cur_var, dv_func, hd_gld)
                  #if cur_var == "dtrix":
                  #  print "cur--- " + str(hd_gld)
                else:
                  # check if current function has target variable
                  if cur_var not in dev_host:
                    for dv_stmt in dv_func:
                      if if_var(cur_var, dv_stmt):
                        param_f.append(get_func_name(dv_func[0]))
                        break
          #remove the global high_precision from high_precision list
          if cur_var in dv_glb:
            dv_glb.remove(cur_var)
            db_glb.append(cur_var)
          if cur_var in db_glb:
            db_glb.remove(cur_var)
            fl_glb.append(cur_var)
          if len(hd_gld) == 0:
            hd_gld.append(cur_var)
          if cur_var in dev_host:
            update_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, "main", cur_var, is_single)
          else:
            #tune the variables for those variables that are related cpu and gpu transfer
            #print "param_f all--- " + str(param_f)
            for d_func in param_f:
              if not d_func.strip():
                continue
              #tune functions that contain the target variables
              update_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, d_func, cur_var, is_single)
           
        # tune variables in specified function    
        else:
          del ret_func[:]
          param_func.clear()
          #variables defined and tuned in current function
          update_use_of_var(stmt_list, func, cur_var, ret_func, param_func)
          update_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, func, cur_var, is_single)
        #precision reduction of variables  
      #with open(sys.argv[2], "r") as stmt_list:
      if is_glb:
        with open(sys.argv[2], "r") as stmt_list:
          inside_func = False
          f_cnt = 0
          #update precision for variables on host side
          del ret_func[:]
          param_func.clear()
          target_host = "main"
          if len(hd_gld) == 2 and is_single==False:
            update_use_of_var(stmt_list, target_host, hd_gld[1], ret_func, param_func)
          elif len(hd_gld) == 1:
            update_use_of_var(stmt_list, target_host, hd_gld[0], ret_func, param_func)
            #update_host_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, dv_def, target_host, hd_gld, is_single)
          ret_func = filter(None, ret_func)
          #if cur_var == "hdd":
          #print "param: " + str(param_func)
          #print "ret: " + str(ret_func)
          #print dv_def
          #print hd_gld

          #print "cur: " + cur_var

        """
        print "cur: " + cur_var
        print "ret: " 
        for fc in ret_func:
          print fc
        print "param: " 
        for fc in param_f:
          print fc
        if dv_def != None:
          print "dv def: " + dv_def 
        print "host of device glable: " 
        for fc in hd_gld:
          print fc
        print "ret: " 
        for fc in ret_func:
          print fc
        print "param func: " 
        for fc in param_func:
          print fc
          print str(param_func[fc])
        print "dv_glb"
        print str(dv_glb)
        print "db_glb"
        print str(db_glb)
        """
        #change the host gdd_real allocation and replacement for device and host variables 
        if cur_var in dev_host:
          p_del = []
          for p_key in param_func:
            if "free" in p_key or "qd2gqd" in p_key or "gqd2qd" in p_key:
              p_del.append(p_key)
          for p_d in p_del:
            del param_func[p_d]
          print param_func
          #param_func.clear()
        #if cur_var == "fluxes":
        #  print "var--" + str(hd_gld)
        #  print hd_gld
        #  print ret_func
        #  print str(param_func)
        run_it = len(hd_gld)/2 
        if run_it > 1:
          for r_t in range(0, run_it):

            with open(sys.argv[2], "r") as stmt_list:
               #update precision for variables on host side
               del ret_func[:]
               param_func.clear()
               target_host = "main"
               update_use_of_var(stmt_list, target_host, hd_gld[1], ret_func, param_func)
            #remove irrelevant param_func and ret_func
            if cur_var in dev_host:
               p_del = []
               for p_key in param_func:
                 if "free" in p_key or "qd2gqd" in p_key or "gqd2qd" in p_key:
                   p_del.append(p_key)
               for p_d in p_del:
                 del param_func[p_d]
            ret_func = filter(None, ret_func)
            #print "var--" + str(hd_gld)
            #print "cur---" + str(param_func)
            update_host_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, dv_def, target_host, hd_gld, is_single)
            hd_gld.remove(hd_gld[0])
            hd_gld.remove(hd_gld[0])
        elif run_it == 1 and is_single==True:
          glb_temp = []
          for r_t in range(0, 2):
            del glb_temp[:]
            glb_temp.append(hd_gld[r_t])
            with open(sys.argv[2], "r") as stmt_list:
               #update precision for variables on host side
               del ret_func[:]
               param_func.clear()
               target_host = "main"
               update_use_of_var(stmt_list, target_host, hd_gld[r_t], ret_func, param_func)
            #remove irrelevant param_func and ret_func
            #print "var--" + str(hd_gld[r_t])
            #print "cur---" + str(param_func)
            if cur_var in dev_host:
               p_del = []
               for p_key in param_func:
                 if "free" in p_key or "qd2gqd" in p_key or "gqd2qd" in p_key:
                   p_del.append(p_key)
               for p_d in p_del:
                 del param_func[p_d]
            ret_func = filter(None, ret_func)
            update_host_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, dv_def, target_host, glb_temp, is_single)

        else:
          if cur_var in dev_host:
            param_func.clear()
          update_host_precision(ret_func, param_func, dv_glb, db_glb, fl_glb, dv_def, target_host, hd_gld, is_single)
        # check and change the function parameters when change precision for device host varibles
        if cur_var in dev_host:
          param_func.clear()
          del ret_func[:]
          recur_param(target_host, param_func, cur_var)
          #print "cur_var: " + cur_var
          #print "param: " + str(param_func)
          #print "param func for host_device: " 
          #for fc in param_func:
          #  print fc
          #  print str(param_func[fc])
          update_precision(ret_func, param_func, dv_glb, db_glb, fl_glb,  "empty", cur_var, is_single)
  
      #if vars == "flux_contribution_nb_density_energy":
      #if cur_var == "variables":
      #if cur_var == "finalVec" and is_single==False:
      #if cur_var == "dtrix" and is_single==False:
      #if cur_var == "d_fv_z" and is_single==False:
      if cur_var == "variables" and is_single==False:
        is_single = True
      #elif cur_var == "finalVec" and is_single==True:
      #elif cur_var == "dtrix" and is_single==True:
      #elif cur_var == "d_fv_z" and is_single==True:
      elif cur_var == "variables" and is_single==True:
        is_single = False
      #if cur_var == "vij" and is_single:
      #if cur_var == "dtrix":
      #if cur_var == "rA_shared_z":
      #if cur_var == "variables":
      is_run = True
      if is_run == True:
        print "\tworkon_var: " + cur_var
        with open(sys.argv[2], "r") as var_f:
          del run_program[:]
          for line in var_f:
            line = line.replace("\n", "");
            run_program.append(line)
        subprocess.call(['rm -f ' + "run_program.run"], shell=True)
        with open("run_program.run", "a") as wline:
          for stmt in run_program:
            #skip the empty line
            if "gettimeofday(&end_t" in stmt:
              wline.write(stmt+"\n")
              if is_single:
                for line in db_cmp_insert:
                  wline.write(line+"\n")
              #elif cur_var == "finalVec" and is_single==False:
              #elif cur_var == "dtrix" and is_single==False:
              #elif cur_var == "d_fv_v" and is_single==False:
              #  for line in cmp_v:
              #    wline.write(line+"\n")
              #elif cur_var == "d_fv_x" and is_single==False:
              #  for line in cmp_x:
              #    wline.write(line+"\n")
              #elif cur_var == "d_fv_y" and is_single==False:
              #  for line in cmp_y:
              #    wline.write(line+"\n")
              elif cur_var == "variables" and is_single==False:
                for line in fl_cmp_insert:
                  wline.write(line+"\n")
              else:
                for line in cmp_insert:
                  wline.write(line+"\n")
            else:
              wline.write(stmt+"\n")
        if cur_var in var_combo:
          skip_factor = var_combo[cur_var]
          skip_cnt = 0
          is_skip_for_var_combo = True
        if int(skip_factor) == int(skip_cnt):
          is_skip_for_var_combo = False
          
      #  #run the translator for the original program
        if is_skip_for_var_combo==False:
          #subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/lavaMD/lava_dd.cu;' + ' cd ' + ' ./cuda_bench/lavaMD; make clean;make; ./lava_dd -boxes1d 24'], shell=True)
          #subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/lud/lud_dd.cu;' + ' cd ' + ' ./cuda_bench/lud; make clean;make; ./lud_dd -s 2048'], shell=True)
          #subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/hotspot/hotspot_dd.cu;' + ' cd ' + ' ./cuda_bench/hotspot; make clean;make; ./hotspot_dd 1024 2 2 ../data/hotspot/temp_1024 ../data/hotspot/power_1024 '], shell=True)
          #subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/gaussian/gaussian_dd.cu;' + ' cd ' + ' ./cuda_bench/gaussian; make clean;make; ./gaussian_dd -s 1024'], shell=True)
          #subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/sp/sp_dd.cu;' + ' cd ' + ' ./cuda_bench/sp; make clean;make; ./sp_dd'], shell=True)
          #subprocess.call(['cd ' + ' ./cuda_bench/sp; make clean;make; ./sp_dd'], shell=True)
          subprocess.call(['cp ' + "run_program.run" + ' ./cuda_bench/cfd/euler3d_dd.cu;'], shell=True)
          subprocess.call(['cd ' + ' ./cuda_bench/cfd; make clean;make; ./euler3d_dd ../data/cfd/fvcorr.domn.193K'], shell=True)
          #subprocess.call(['cd ' + ' ./cuda_bench/cfd; make clean;make; ./euler3d_dd ../data/cfd/fvcorr.domn.097K'], shell=True)
        else:
          skip_cnt += 1
        #if cur_var == "normals" and is_single==True:
        #if cur_var == "fu" and is_single==True:
        #if cur_var == "dtrix" and is_single:
        #if cur_var == "vij" and is_single:
        #if cur_var == "fluxes":
        #if cur_var == "variables":
        if cur_var == "variables" and is_single==False:
          exit()

#subprocess.call(['cp ' + str(sys.argv[2]) + ' ./cuda_bench/cfd/euler3d_dd.cu'], shell=True)
#subprocess.call(['cd ' + ' ./cuda_bench/cfd; make clean;make; ./euler3d_dd ../data/cfd/fvcorr.domn.097K'], shell=True)

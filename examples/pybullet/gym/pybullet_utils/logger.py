import pybullet_utils.mpi_util as MPIUtil
"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_file() to start logging to a 
tab-separated-values file (some_file_name.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""

import os.path as osp, shutil, time, atexit, os, subprocess
try:
  import tensorflow.compat.v1 as tf
except Exception:
  import tensorflow as tf


class Logger:

  def print2(str):
    if (MPIUtil.is_root_proc()):
      print(str)
    return

  def __init__(self):
    self.output_file = None
    self.first_row = True
    self.log_headers = []
    self.log_current_row = {}
    self._dump_str_template = ""
    self._tb_dir = ""  # 로그를 저장할 폴더 경로
    self.summary_writer = None  # 로그를 기록할 FileWriter
    return

  def configure_tensorboard(self, log_dir):  # 로그를 저장하는 데 필요한 graph와 session, 그리고 기타 등등 필요한 것들을 Logger의 property로 만들어두는 함수
    self.tb_first_row = True
    self.summary_graph = tf.Graph()
    self.summary_sess = tf.Session(graph = self.summary_graph)
    self.summary_writer = tf.summary.FileWriter(log_dir)
    self.key2idx = {}
    self.feed_dict = {}
    self.placeholders = []
    self.scalars = []
    return

  def log_tb(self, key, val):  # 값을 받아 로그로 기록해두는 함수
    if (MPIUtil.is_root_proc()):
      if self.tb_first_row:
        if key not in self.key2idx:
          with self.summary_graph.as_default():
            self.placeholders.append(tf.placeholder(dtype = tf.float32))
            self.scalars.append(tf.summary.scalar(name=key, tensor=self.placeholders[-1]))
          self.key2idx[key] = len(self.placeholders) - 1
      self.feed_dict[self.placeholders[self.key2idx[key]]] = val
    return

  def tb_add_summary(self, iter):  # 매 iteration마다 기록된 로그를 묶어 summary로 저장하는 함수
    if (MPIUtil.is_root_proc()):
      if self.tb_first_row:
        self.tb_first_row = False
        with self.summary_graph.as_default():
          self.summaries = tf.summary.merge_all()
        # self.summaries = tf.summary.merge_all()
      summary = self.summary_sess.run(self.summaries, feed_dict=self.feed_dict)
      self.summary_writer.add_summary(summary, global_step=iter)
    return

  def reset(self):
    self.first_row = True
    self.log_headers = []
    self.log_current_row = {}
    if self.output_file is not None:
      self.output_file = open(self.output_path, 'w')
    return

  def configure_output_file(self, filename=None):
    """
        Set output directory to d, or to /tmp/somerandomnumber if d is None
        """
    self.first_row = True
    self.log_headers = []
    self.log_current_row = {}

    self.output_path = filename or "output/log_%i.txt" % int(time.time())

    out_dir = os.path.dirname(self.output_path)
    if not os.path.exists(out_dir) and MPIUtil.is_root_proc():
      os.makedirs(out_dir)

    if (MPIUtil.is_root_proc()):
      self.output_file = open(self.output_path, 'w')
      assert osp.exists(self.output_path)
      atexit.register(self.output_file.close)

      Logger.print2("Logging data to " + self.output_file.name)
    return

  def log_tabular(self, key, val):
    """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
    if self.first_row and key not in self.log_headers:
      self.log_headers.append(key)
    else:
      assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
    self.log_current_row[key] = val
    return

  def get_num_keys(self):
    return len(self.log_headers)

  def print_tabular(self):
    """
        Print all of the diagnostics from the current iteration
        """
    if (MPIUtil.is_root_proc()):
      vals = []
      Logger.print2("-" * 37)
      for key in self.log_headers:
        val = self.log_current_row.get(key, "")
        if isinstance(val, float):
          valstr = "%8.3g" % val
        elif isinstance(val, int):
          valstr = str(val)
        else:
          valstr = val
        Logger.print2("| %15s | %15s |" % (key, valstr))
        vals.append(val)
      Logger.print2("-" * 37)
    return

  def dump_tabular(self):
    """
        Write all of the diagnostics from the current iteration
        """
    if (MPIUtil.is_root_proc()):
      if (self.first_row):
        self._dump_str_template = self._build_str_template()

      vals = []
      for key in self.log_headers:
        val = self.log_current_row.get(key, "")
        vals.append(val)

      if self.output_file is not None:
        if self.first_row:
          header_str = self._dump_str_template.format(*self.log_headers)
          self.output_file.write(header_str + "\n")

        val_str = self._dump_str_template.format(*map(str, vals))
        self.output_file.write(val_str + "\n")
        self.output_file.flush()

    self.log_current_row.clear()
    self.first_row = False
    return

  def _build_str_template(self):
    num_keys = self.get_num_keys()
    template = "{:<25}" * num_keys
    return template

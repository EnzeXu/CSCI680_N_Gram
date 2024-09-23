public class Hadoop3MRPlatform extends BaseHadoopPlatform < JobConf > { private static final Logger LOG = LoggerFactory . getLogger ( Hadoop3MRPlatform . class ) ; private transient static MiniDFSCluster dfs ; private transient static MiniMRClientCluster mr ; public Hadoop3MRPlatform ( ) { } @ Override public String getName ( ) { return "hadoop3-mr1" ; } @ Override public FlowConnector getFlowConnector ( Map < Object , Object > properties ) { return new Hadoop3MRFlowConnector ( properties ) ; } @ Override public void setNumMapTasks ( Map < Object , Object > properties , int numMapTasks ) { properties . put ( "mapreduce . job . maps" , Integer . toString ( numMapTasks ) ) ; } @ Override public void setNumReduceTasks ( Map < Object , Object > properties , int numReduceTasks ) { properties . put ( "mapreduce . job . reduces" , Integer . toString ( numReduceTasks ) ) ; } @ Override public Integer getNumMapTasks ( Map < Object , Object > properties ) { if ( properties . get ( "mapreduce . job . maps" ) == null ) return null ; return Integer . parseInt ( properties . get ( "mapreduce . job . maps" ) . toString ( ) ) ; } @ Override public Integer getNumReduceTasks ( Map < Object , Object > properties ) { if ( properties . get ( "mapreduce . job . reduces" ) == null ) return null ; return Integer . parseInt ( properties . get ( "mapreduce . job . reduces" ) . toString ( ) ) ; } public JobConf getConfiguration ( ) { return new JobConf ( configuration ) ; } @ Override public Tap getDistCacheTap ( Hfs parent ) { return new DistCacheTap ( parent ) ; } @ Override public FlowProcess getFlowProcess ( ) { return new HadoopFlowProcess ( FlowSession . NULL , getConfiguration ( ) , true ) ; } @ Override public synchronized void setUp ( ) throws IOException { if ( configuration != null ) return ; if ( !isUseCluster ( ) ) { LOG . info ( "not using cluster" ) ; configuration = new JobConf ( ) ; configuration . set ( "fs . defaultFS" , "file : configuration . set ( "mapreduce . framework . name" , "local" ) ; configuration . set ( "mapreduce . jobtracker . staging . root . dir" , System . getProperty ( "user . dir" ) + "/" + "build/tmp/cascading/staging" ) ; String stagingDir = configuration . get ( "mapreduce . jobtracker . staging . root . dir" ) ; if ( Util . isEmpty ( stagingDir ) ) configuration . set ( "mapreduce . jobtracker . staging . root . dir" , System . getProperty ( "user . dir" ) + "/build/tmp/cascading/staging" ) ; fileSys = FileSystem . get ( configuration ) ; } else { LOG . info ( "using cluster" ) ; if ( Util . isEmpty ( System . getProperty ( "hadoop . log . dir" ) ) ) System . setProperty ( "hadoop . log . dir" , "build/test/log" ) ; if ( Util . isEmpty ( System . getProperty ( "hadoop . tmp . dir" ) ) ) System . setProperty ( "hadoop . tmp . dir" , "build/test/tmp" ) ; new File ( System . getProperty ( "hadoop . log . dir" ) ) . mkdirs ( ) ; JobConf conf = new JobConf ( ) ; if ( getApplicationJar ( ) != null ) { LOG . info ( "using a remote cluster with jar : { } " , getApplicationJar ( ) ) ; configuration = conf ; ( ( JobConf ) configuration ) . setJar ( getApplicationJar ( ) ) ; if ( !Util . isEmpty ( System . getProperty ( "fs . default . name" ) ) ) { LOG . info ( "using { } = { } " , "fs . default . name" , System . getProperty ( "fs . default . name" ) ) ; configuration . set ( "fs . default . name" , System . getProperty ( "fs . default . name" ) ) ; } if ( !Util . isEmpty ( System . getProperty ( "mapred . job . tracker" ) ) ) { LOG . info ( "using { } = { } " , "mapred . job . tracker" , System . getProperty ( "mapred . job . tracker" ) ) ; configuration . set ( "mapred . job . tracker" , System . getProperty ( "mapred . job . tracker" ) ) ; } if ( !Util . isEmpty ( System . getProperty ( "fs . defaultFS" ) ) ) { LOG . info ( "using { } = { } " , "fs . defaultFS" , System . getProperty ( "fs . defaultFS" ) ) ; configuration . set ( "fs . defaultFS" , System . getProperty ( "fs . defaultFS" ) ) ; } if ( !Util . isEmpty ( System . getProperty ( "yarn . resourcemanager . address" ) ) ) { LOG . info ( "using { } = { } " , "yarn . resourcemanager . address" , System . getProperty ( "yarn . resourcemanager . address" ) ) ; configuration . set ( "yarn . resourcemanager . address" , System . getProperty ( "yarn . resourcemanager . address" ) ) ; } if ( !Util . isEmpty ( System . getProperty ( "mapreduce . jobhistory . address" ) ) ) { LOG . info ( "using { } = { } " , "mapreduce . jobhistory . address" , System . getProperty ( "mapreduce . jobhistory . address" ) ) ; configuration . set ( "mapreduce . jobhistory . address" , System . getProperty ( "mapreduce . jobhistory . address" ) ) ; } configuration . set ( "mapreduce . job . user . classpath . first" , "true" ) ; configuration . set ( "mapreduce . user . classpath . first" , "true" ) ; configuration . set ( "mapreduce . framework . name" , "yarn" ) ; fileSys = FileSystem . get ( configuration ) ; } else { conf . setBoolean ( "yarn . is . minicluster" , true ) ; conf . setBoolean ( "yarn . app . mapreduce . am . job . node-blacklisting . enable" , false ) ; dfs = new MiniDFSCluster ( conf , 4 , true , null ) ; fileSys = dfs . getFileSystem ( ) ; FileSystem . setDefaultUri ( conf , fileSys . getUri ( ) ) ; mr = MiniMRClientClusterFactory . create ( this . getClass ( ) , 4 , conf ) ; configuration = mr . getConfig ( ) ; } configuration . set ( "mapred . child . java . opts" , "-Xmx512m" ) ; configuration . setInt ( "mapreduce . job . jvm . numtasks" , -1 ) ; configuration . setInt ( "mapreduce . client . completion . pollinterval" , 50 ) ; configuration . setInt ( "mapreduce . client . progressmonitor . pollinterval" , 50 ) ; configuration . setBoolean ( "mapreduce . map . speculative" , false ) ; configuration . setBoolean ( "mapreduce . reduce . speculative" , false ) ; } configuration . setInt ( "mapreduce . job . maps" , numMappers ) ; configuration . setInt ( "mapreduce . job . reduces" , numReducers ) ; Map < Object , Object > globalProperties = getGlobalProperties ( ) ; if ( logger != null ) globalProperties . put ( "log4j . logger" , logger ) ; FlowProps . setJobPollingInterval ( globalProperties , 10 ) ; Hadoop3MRPlanner . copyProperties ( configuration , globalProperties ) ; Hadoop3MRPlanner . copyConfiguration ( properties , configuration ) ; } }
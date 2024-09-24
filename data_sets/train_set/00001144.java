public class Hadoop3TezPlatform extends BaseHadoopPlatform<TezConfiguration> { private static final Logger LOG = LoggerFactory.getLogger( Hadoop3TezPlatform.class ); private transient static MiniDFSCluster miniDFSCluster; private transient static MiniTezCluster miniTezCluster; private transient static SecurityManager securityManager; private transient ApplicationHistoryServer yarnHistoryServer; public Hadoop3TezPlatform() { this.numGatherPartitions = 1; } @Override public String getName() { return "hadoop3-tez"; } @Override public FlowConnector getFlowConnector( Map<Object, Object> properties ) { return new Hadoop3TezFlowConnector( properties ); } @Override public void setNumGatherPartitionTasks( Map<Object, Object> properties, int numGatherPartitions ) { properties.put( FlowRuntimeProps.GATHER_PARTITIONS, Integer.toString( numGatherPartitions ) ); } @Override public Integer getNumGatherPartitionTasks( Map<Object, Object> properties ) { if( properties.get( FlowRuntimeProps.GATHER_PARTITIONS ) == null ) return null; return Integer.parseInt( properties.get( FlowRuntimeProps.GATHER_PARTITIONS ).toString() ); } public TezConfiguration getConfiguration() { return new TezConfiguration( configuration ); } @Override public Tap getDistCacheTap( Hfs parent ) { return new DistCacheTap( parent ); } @Override public FlowProcess getFlowProcess() { return new Hadoop3TezFlowProcess( FlowSession.NULL, null, getConfiguration() ); } @Override public boolean isMapReduce() { return false; } @Override public boolean isDAG() { return true; } @Override public synchronized void setUp() throws IOException { if( configuration != null ) return; if( !isUseCluster() ) { LOG.info( "not using cluster" ); configuration = new Configuration(); configuration.setInt( FlowRuntimeProps.GATHER_PARTITIONS, getNumGatherPartitions() ); configuration.set( TezConfiguration.TEZ_LOCAL_MODE, "true" ); configuration.set( "fs.defaultFS", "file: configuration.set( "tez.runtime.optimize.local.fetch", "true" ); configuration.setInt( "tez.am.inline.task.execution.max-tasks", 3 ); configuration.set( TezConfiguration.TEZ_IGNORE_LIB_URIS, "true" ); configuration.setInt( YarnConfiguration.DEBUG_NM_DELETE_DELAY_SEC, -1 ); configuration.set( TezConfiguration.TEZ_GENERATE_DEBUG_ARTIFACTS, "true" ); configuration.set( "tez.am.mode.session", "true" ); if( !Util.isEmpty( System.getProperty( "hadoop.tmp.dir" ) ) ) configuration.set( "hadoop.tmp.dir", System.getProperty( "hadoop.tmp.dir" ) ); else configuration.set( "hadoop.tmp.dir", "build/test/tmp" ); fileSys = FileSystem.get( configuration ); } else { LOG.info( "using cluster" ); if( Util.isEmpty( System.getProperty( "hadoop.log.dir" ) ) ) System.setProperty( "hadoop.log.dir", "build/test/log" ); if( Util.isEmpty( System.getProperty( "hadoop.tmp.dir" ) ) ) System.setProperty( "hadoop.tmp.dir", "build/test/tmp" ); new File( System.getProperty( "hadoop.log.dir" ) ).mkdirs(); new File( System.getProperty( "hadoop.tmp.dir" ) ).mkdirs(); Configuration defaultConf = new Configuration(); defaultConf.setInt( FlowRuntimeProps.GATHER_PARTITIONS, getNumGatherPartitions() ); defaultConf.setInt( YarnConfiguration.DEBUG_NM_DELETE_DELAY_SEC, -1 ); defaultConf.setInt( YarnConfiguration.RM_AM_MAX_ATTEMPTS, 1 ); defaultConf.setBoolean( TezConfiguration.TEZ_AM_NODE_BLACKLISTING_ENABLED, false ); defaultConf.set( MiniDFSCluster.HDFS_MINIDFS_BASEDIR, System.getProperty( "hadoop.tmp.dir" ) ); miniDFSCluster = new MiniDFSCluster.Builder( defaultConf ) .numDataNodes( 4 ) .format( true ) .racks( null ) .build(); fileSys = miniDFSCluster.getFileSystem(); Configuration tezConf = new Configuration( defaultConf ); tezConf.set( "fs.defaultFS", fileSys.getUri().toString() ); tezConf.set( MRJobConfig.MR_AM_STAGING_DIR, "/apps_staging_dir" ); miniTezCluster = new MiniTezCluster( getClass().getName(), 4, 1, 1 ); miniTezCluster.init( tezConf ); miniTezCluster.start(); configuration = miniTezCluster.getConfig(); if( setTimelineStore( configuration ) ) { configuration.set( TezConfiguration.TEZ_HISTORY_LOGGING_SERVICE_CLASS, ATSHistoryLoggingService.class.getName() ); configuration.setBoolean( YarnConfiguration.TIMELINE_SERVICE_ENABLED, true ); configuration.set( YarnConfiguration.TIMELINE_SERVICE_ADDRESS, "localhost:10200" ); configuration.set( YarnConfiguration.TIMELINE_SERVICE_WEBAPP_ADDRESS, "localhost:8188" ); configuration.set( YarnConfiguration.TIMELINE_SERVICE_WEBAPP_HTTPS_ADDRESS, "localhost:8190" ); yarnHistoryServer = new ApplicationHistoryServer(); yarnHistoryServer.init( configuration ); yarnHistoryServer.start(); } } configuration.setInt( TezConfiguration.TEZ_AM_MAX_APP_ATTEMPTS, 1 ); configuration.setInt( TezConfiguration.TEZ_AM_TASK_MAX_FAILED_ATTEMPTS, 1 ); configuration.setInt( TezConfiguration.TEZ_AM_MAX_TASK_FAILURES_PER_NODE, 1 ); Map<Object, Object> globalProperties = getGlobalProperties(); if( logger != null ) globalProperties.put( "log4j.logger", logger ); FlowProps.setJobPollingInterval( globalProperties, 10 ); Hadoop3TezPlanner.copyProperties( configuration, globalProperties ); Hadoop3TezPlanner.copyConfiguration( properties, configuration ); ExitUtil.disableSystemExit(); } protected boolean setTimelineStore( Configuration configuration ) { try { Class<?> target = Util.loadClass( "org.apache.hadoop.yarn.server.timeline.TimelineStore" ); Class<?> type = Util.loadClass( "org.apache.hadoop.yarn.server.timeline.MemoryTimelineStore" ); configuration.setClass( YarnConfiguration.TIMELINE_SERVICE_STORE, type, target ); try { Util.loadClass( "org.apache.hadoop.yarn.api.records.timeline.TimelineDomain" ); } catch( CascadingException exception ) { configuration.setBoolean( TezConfiguration.TEZ_AM_ALLOW_DISABLED_TIMELINE_DOMAINS, true ); } return true; } catch( CascadingException exception ) { try { Class<?> target = Util.loadClass( "org.apache.hadoop.yarn.server.applicationhistoryservice.timeline.TimelineStore" ); Class<?> type = Util.loadClass( "org.apache.hadoop.yarn.server.applicationhistoryservice.timeline.MemoryTimelineStore" ); configuration.setClass( YarnConfiguration.TIMELINE_SERVICE_STORE, type, target ); configuration.setBoolean( TezConfiguration.TEZ_AM_ALLOW_DISABLED_TIMELINE_DOMAINS, true ); return true; } catch( CascadingException ignore ) { return false; } } } private static class ExitTrappedException extends SecurityException { } private static void forbidSystemExitCall() { if( securityManager != null ) return; securityManager = new SecurityManager() { public void checkPermission( Permission permission ) { if( !"exitVM".equals( permission.getName() ) ) return; StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace(); for( StackTraceElement stackTraceElement : stackTrace ) LOG.warn( "exit vm trace: {}", stackTraceElement ); throw new ExitTrappedException(); } }; System.setSecurityManager( securityManager ); } private static void enableSystemExitCall() { securityManager = null; System.setSecurityManager( null ); } }
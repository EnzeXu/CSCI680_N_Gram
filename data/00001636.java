public class HadoopFlow extends BaseFlow < JobConf > { private static Thread hdfsShutdown = null ; private static ShutdownUtil . Hook shutdownHook ; private transient JobConf jobConf ; private boolean preserveTemporaryFiles = false ; private transient Map < Path , Path > syncPaths ; protected HadoopFlow ( ) { } static boolean getPreserveTemporaryFiles ( Map < Object , Object > properties ) { return Boolean . parseBoolean ( PropertyUtil . getProperty ( properties , PRESERVE_TEMPORARY_FILES , "false" ) ) ; } static int getMaxConcurrentSteps ( JobConf jobConf ) { return jobConf . getInt ( MAX_CONCURRENT_STEPS , 0 ) ; } protected HadoopFlow ( PlatformInfo platformInfo , Map < Object , Object > properties , JobConf jobConf , String name , Map < String , String > flowDescriptor ) { super ( platformInfo , properties , jobConf , name , flowDescriptor ) ; initFromProperties ( properties ) ; } public HadoopFlow ( PlatformInfo platformInfo , Map < Object , Object > properties , JobConf jobConf , FlowDef flowDef ) { super ( platformInfo , properties , jobConf , flowDef ) ; initFromProperties ( properties ) ; } @ Override protected void initFromProperties ( Map < Object , Object > properties ) { super . initFromProperties ( properties ) ; preserveTemporaryFiles = getPreserveTemporaryFiles ( properties ) ; } protected void initConfig ( Map < Object , Object > properties , JobConf parentConfig ) { if ( properties != null ) parentConfig = createConfig ( properties , parentConfig ) ; if ( parentConfig == null ) return ; jobConf = HadoopUtil . copyJobConf ( parentConfig ) ; jobConf . set ( "fs . http . impl" , HttpFileSystem . class . getName ( ) ) ; jobConf . set ( "fs . https . impl" , HttpFileSystem . class . getName ( ) ) ; syncPaths = HadoopMRUtil . addToClassPath ( jobConf , getClassPath ( ) ) ; } @ Override protected void setConfigProperty ( JobConf config , Object key , Object value ) { if ( value instanceof Class || value instanceof JobConf || value == null ) return ; config . set ( key . toString ( ) , value . toString ( ) ) ; } @ Override protected JobConf newConfig ( JobConf defaultConfig ) { return defaultConfig == null ? new JobConf ( ) : HadoopUtil . copyJobConf ( defaultConfig ) ; } @ ProcessConfiguration @ Override public JobConf getConfig ( ) { if ( jobConf == null ) initConfig ( null , new JobConf ( ) ) ; return jobConf ; } @ Override public JobConf getConfigCopy ( ) { return HadoopUtil . copyJobConf ( getConfig ( ) ) ; } @ Override public Map < Object , Object > getConfigAsProperties ( ) { return HadoopUtil . createProperties ( getConfig ( ) ) ; } public String getProperty ( String key ) { return getConfig ( ) . get ( key ) ; } @ Override public FlowProcess < JobConf > getFlowProcess ( ) { return new HadoopFlowProcess ( getFlowSession ( ) , getConfig ( ) ) ; } public boolean isPreserveTemporaryFiles ( ) { return preserveTemporaryFiles ; } @ Override protected void internalStart ( ) { try { copyToDistributedCache ( ) ; deleteSinksIfReplace ( ) ; deleteTrapsIfReplace ( ) ; deleteCheckpointsIfReplace ( ) ; } catch ( IOException exception ) { throw new FlowException ( "unable to delete sinks" , exception ) ; } registerHadoopShutdownHook ( ) ; } protected void registerHadoopShutdownHook ( ) { registerHadoopShutdownHook ( this ) ; } protected void copyToDistributedCache ( ) { HadoopUtil . syncPaths ( jobConf , syncPaths , true ) ; } @ Override public boolean stepsAreLocal ( ) { return HadoopUtil . isLocal ( getConfig ( ) ) ; } private void cleanTemporaryFiles ( boolean stop ) { if ( stop ) return ; for ( FlowStep < JobConf > step : getFlowSteps ( ) ) ( ( BaseFlowStep < JobConf > ) step ) . clean ( ) ; } private static synchronized void registerHadoopShutdownHook ( Flow flow ) { if ( !flow . isStopJobsOnExit ( ) ) return ; if ( shutdownHook != null ) return ; getHdfsShutdownHook ( ) ; shutdownHook = new ShutdownUtil . Hook ( ) { @ Override public Priority priority ( ) { return Priority . LAST ; } @ Override public void execute ( ) { callHdfsShutdownHook ( ) ; } } ; ShutdownUtil . addHook ( shutdownHook ) ; } private synchronized static void callHdfsShutdownHook ( ) { if ( hdfsShutdown != null ) hdfsShutdown . start ( ) ; } private synchronized static void getHdfsShutdownHook ( ) { if ( hdfsShutdown == null ) hdfsShutdown = HadoopUtil . getHDFSShutdownHook ( ) ; } protected void internalClean ( boolean stop ) { if ( !isPreserveTemporaryFiles ( ) ) cleanTemporaryFiles ( stop ) ; } protected void internalShutdown ( ) { } protected int getMaxNumParallelSteps ( ) { return stepsAreLocal ( ) ? 1 : getMaxConcurrentSteps ( getConfig ( ) ) ; } @ Override protected long getTotalSliceCPUMilliSeconds ( ) { long counterValue = flowStats . getCounterValue ( "org . apache . hadoop . mapreduce . TaskCounter" , "CPU_MILLISECONDS" ) ; if ( counterValue == 0 ) return -1 ; return counterValue ; } }
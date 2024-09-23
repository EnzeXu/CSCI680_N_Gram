public class TezStatsUtil { private static final Logger LOG = LoggerFactory.getLogger( TezStatsUtil.class ); public static final Set<StatusGetOpts> STATUS_GET_COUNTERS = EnumSet.of( StatusGetOpts.GET_COUNTERS ); static Class<DAGClient> timelineClientClass = null; public static final String TIMELINE_CLIENT_CLASS = "cascading.stats.tez.util.TezTimelineClient"; public static String getPlatformVersion() { PlatformInfo tez = HadoopUtil.getPlatformInfo( DAG.class, null, "Tez" ); if( tez == null || tez.version == null ) return "unknown"; return tez.version; } private static boolean loadClass() { if( timelineClientClass != null ) return true; try { timelineClientClass = (Class<DAGClient>) Thread.currentThread().getContextClassLoader().loadClass( TIMELINE_CLIENT_CLASS ); return true; } catch( ClassNotFoundException exception ) { LOG.error( "'" + YarnConfiguration.TIMELINE_SERVICE_ENABLED + "' is enabled, yet unable to load Tez YARN timeline client class: {}, ensure these dependencies are in your local CLASSPATH: tez-yarn-timeline-history, org.apache.tez:tez-yarn-timeline-history or org.apache.tez:tez-yarn-timeline-history-with-acls", TIMELINE_CLIENT_CLASS, exception ); } return false; } public static DAGStatus getDagStatusWithCounters( DAGClient dagClient ) { if( dagClient == null ) return null; try { return dagClient.getDAGStatus( STATUS_GET_COUNTERS ); } catch( IOException | TezException exception ) { throw new CascadingException( "unable to get counters from dag client", exception ); } } public static DAGClient createTimelineClient( DAGClient dagClient ) { if( dagClient == null ) return null; if( !loadClass() ) return null; Class[] types = new Class[]{ ApplicationId.class, String.class, TezConfiguration.class, FrameworkClient.class, DAGClient.class }; ApplicationId appId = Util.returnInstanceFieldIfExistsSafe( dagClient, "appId" ); String dagId = Util.returnInstanceFieldIfExistsSafe( dagClient, "dagId" ); TezConfiguration conf = Util.returnInstanceFieldIfExistsSafe( dagClient, "conf" ); FrameworkClient frameworkClient = Util.returnInstanceFieldIfExistsSafe( dagClient, "frameworkClient" ); Object[] parameters = new Object[]{ appId, dagId, conf, frameworkClient, dagClient }; try { return Util.invokeConstructor( timelineClientClass, parameters, types ); } catch( CascadingException exception ) { Throwable cause = exception.getCause(); if( cause instanceof ReflectiveOperationException && cause.getCause() instanceof TezException ) LOG.warn( "unable to construct timeline server client", cause.getCause() ); else if( cause instanceof ReflectiveOperationException && cause.getCause() instanceof NoSuchMethodError ) LOG.warn( "unable to construct timeline server client, check for compatible Tez version, current: {}", getPlatformVersion(), cause.getCause() ); else LOG.warn( "unable to construct timeline server client", exception ); } return null; } public static String getTrackingURL( TezClient tezClient, DAGClient dagClient ) { if( tezClient == null || dagClient == null ) return null; try { ApplicationId applicationId = tezClient.getAppMasterApplicationId(); FrameworkClient frameworkClient = getFrameworkClient( dagClient ); if( frameworkClient == null ) { LOG.info( "unable to get framework client" ); return null; } ApplicationReport report = frameworkClient.getApplicationReport( applicationId ); if( report != null ) return report.getTrackingUrl(); } catch( YarnException | IOException exception ) { LOG.info( "unable to get tracking url" ); LOG.debug( "exception retrieving application report", exception ); } return null; } private static FrameworkClient getFrameworkClient( DAGClient dagClient ) { if( dagClient instanceof TezTimelineClient ) return ( (TezTimelineClient) dagClient ).getFrameworkClient(); return Util.returnInstanceFieldIfExistsSafe( dagClient, "frameworkClient" ); } }
public class TezTimelineClient extends DAGClient implements TimelineClient { private final String dagId ; private final FrameworkClient frameworkClient ; private final DAGClient dagClient ; public TezTimelineClient ( ApplicationId appId , String dagId , TezConfiguration conf , FrameworkClient frameworkClient , DAGClient dagClient ) throws TezException { this . dagId = dagId ; this . frameworkClient = frameworkClient ; this . dagClient = dagClient ; } public DAGClient getDAGClient ( ) { return dagClient ; } public FrameworkClient getFrameworkClient ( ) { return frameworkClient ; } @ Override public DAGStatus getDAGStatus ( @ Nullable Set < StatusGetOpts > statusOptions ) throws IOException , TezException { return dagClient . getDAGStatus ( statusOptions ) ; } @ Override public DAGStatus getDAGStatus ( @ Nullable Set < StatusGetOpts > statusOptions , long timeout ) throws IOException , TezException { return dagClient . getDAGStatus ( statusOptions , timeout ) ; } @ Override public VertexStatus getVertexStatus ( String vertexName , Set < StatusGetOpts > statusOptions ) throws IOException , TezException { return dagClient . getVertexStatus ( vertexName , statusOptions ) ; } @ Override public void tryKillDAG ( ) throws IOException , TezException { dagClient . tryKillDAG ( ) ; } @ Override public DAGStatus waitForCompletion ( ) throws IOException , TezException , InterruptedException { return dagClient . waitForCompletion ( ) ; } @ Override public void close ( ) throws IOException { dagClient . close ( ) ; } @ Override public DAGStatus waitForCompletionWithStatusUpdates ( @ Nullable Set < StatusGetOpts > statusOpts ) throws IOException , TezException , InterruptedException { return dagClient . waitForCompletionWithStatusUpdates ( statusOpts ) ; } @ Override public String getWebUIAddress ( ) throws IOException , TezException { return dagClient . getWebUIAddress ( ) ; } @ Override public String getSessionIdentifierString ( ) { return dagClient . getSessionIdentifierString ( ) ; } @ Override public String getDagIdentifierString ( ) { return dagClient . getDagIdentifierString ( ) ; } @ Override public String getExecutionContext ( ) { return dagClient . getExecutionContext ( ) ; } @ Override public String getVertexID ( String vertexName ) throws IOException , TezException { throw new TezException ( "reporting API is temporarily disabled on TEZ-3369 implementation" ) ; } @ Override public Iterator < TaskStatus > getVertexChildren ( String vertexID , int limit , String startTaskID ) throws IOException , TezException { throw new TezException ( "reporting API is temporarily disabled on TEZ-3369 implementation" ) ; } @ Override public TaskStatus getVertexChild ( String taskID ) throws TezException { throw new TezException ( "reporting API is temporarily disabled on TEZ-3369 implementation" ) ; } @ Override protected ApplicationReport getApplicationReportInternal ( ) { return null ; } }
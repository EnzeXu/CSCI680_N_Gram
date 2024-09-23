public class MultiMapReduceFlow extends BaseMapReduceFlow { private Map<String, Tap> tapCache = new HashMap<>(); private List<MapReduceFlowStep> queuedSteps = new LinkedList<>(); private volatile boolean completeCalled = false; private final Object lock = new Object(); public MultiMapReduceFlow( String name, JobConf jobConf, JobConf... jobConfs ) { this( HadoopUtil.getPlatformInfo( JobConf.class, "org/apache/hadoop", "Hadoop MR" ), new Properties(), name ); initializeFrom( asList( jobConf, jobConfs ) ); } public MultiMapReduceFlow( Map<Object, Object> properties, String name, JobConf jobConf, JobConf... jobConfs ) { this( HadoopUtil.getPlatformInfo( JobConf.class, "org/apache/hadoop", "Hadoop MR" ), properties, name, null ); initializeFrom( asList( jobConf, jobConfs ) ); } public MultiMapReduceFlow( Map<Object, Object> properties, String name, Map<String, String> flowDescriptor, JobConf jobConf, JobConf... jobConfs ) { this( HadoopUtil.getPlatformInfo( JobConf.class, "org/apache/hadoop", "Hadoop MR" ), properties, name, flowDescriptor ); initializeFrom( asList( jobConf, jobConfs ) ); } public MultiMapReduceFlow( Map<Object, Object> properties, String name, Map<String, String> flowDescriptor, boolean stopJobsOnExit, JobConf jobConf, JobConf... jobConfs ) { this( HadoopUtil.getPlatformInfo( JobConf.class, "org/apache/hadoop", "Hadoop MR" ), properties, name, flowDescriptor ); this.stopJobsOnExit = stopJobsOnExit; initializeFrom( asList( jobConf, jobConfs ) ); } protected MultiMapReduceFlow( PlatformInfo platformInfo, Map<Object, Object> properties, String name ) { this( platformInfo, properties, name, null ); } protected MultiMapReduceFlow( PlatformInfo platformInfo, Map<Object, Object> properties, String name, Map<String, String> flowDescriptor ) { super( platformInfo, properties, name, flowDescriptor, false ); } protected void initializeFrom( List<JobConf> jobConfs ) { List<MapReduceFlowStep> steps = new ArrayList<>(); for( JobConf jobConf : jobConfs ) steps.add( createMapReduceFlowStep( jobConf ) ); updateWithFlowSteps( steps ); } protected MapReduceFlowStep createMapReduceFlowStep( JobConf jobConf ) { return new MapReduceFlowStep( this, jobConf ); } public void notifyComplete() { completeCalled = true; synchronized( lock ) { lock.notifyAll(); } } @Override public void complete() { notifyComplete(); super.complete(); } @Override protected boolean spawnSteps() throws InterruptedException, ExecutionException { while( !stop && throwable == null ) { if( !blockingContinuePollingSteps() ) return true; if( isInfoEnabled() ) { logInfo( "updated" ); for( Tap source : getSourcesCollection() ) logInfo( " source: " + source ); for( Tap sink : getSinksCollection() ) logInfo( " sink: " + sink ); } if( !super.spawnSteps() ) return false; } return true; } protected boolean blockingContinuePollingSteps() { synchronized( lock ) { while( queuedSteps.isEmpty() && !completeCalled ) { try { lock.wait(); } catch( InterruptedException exception ) { } } updateWithFlowSteps( queuedSteps ).clear(); } if( getEligibleJobsSize() != 0 ) return true; return !completeCalled; } @Override protected Tap createTap( JobConf jobConf, Path path, SinkMode sinkMode ) { Tap tap = tapCache.get( path.toString() ); if( tap == null ) { tap = super.createTap( jobConf, path, sinkMode ); tapCache.put( path.toString(), tap ); } return tap; } public void attachFlowStep( JobConf jobConf ) { if( completeCalled ) throw new IllegalStateException( "cannot attach new FlowStep after complete() has been called" ); addFlowStep( createMapReduceFlowStep( jobConf ) ); } protected void addFlowStep( MapReduceFlowStep flowStep ) { synchronized( lock ) { queuedSteps.add( flowStep ); lock.notifyAll(); } } protected FlowStepGraph getOrCreateFlowStepGraph() { FlowStepGraph flowStepGraph = getFlowStepGraph(); if( flowStepGraph == null ) { flowStepGraph = new FlowStepGraph(); setFlowStepGraph( flowStepGraph ); } return flowStepGraph; } protected Collection<MapReduceFlowStep> updateWithFlowSteps( Collection<MapReduceFlowStep> flowSteps ) { if( flowSteps.isEmpty() ) return flowSteps; FlowStepGraph flowStepGraph = getOrCreateFlowStepGraph(); updateFlowStepGraph( flowStepGraph, flowSteps ); setFlowElementGraph( asFlowElementGraph( platformInfo, flowStepGraph ) ); removeListeners( getSourcesCollection() ); removeListeners( getSinksCollection() ); removeListeners( getTrapsCollection() ); setSources( flowStepGraph.getSourceTapsMap() ); setSinks( flowStepGraph.getSinkTapsMap() ); setTraps( flowStepGraph.getTrapsMap() ); initSteps(); if( flowStats == null ) flowStats = createPrepareFlowStats(); if( !isJobsMapInitialized() ) initializeNewJobsMap(); else updateJobsMap(); initializeChildStats(); return flowSteps; } protected FlowStepGraph updateFlowStepGraph( FlowStepGraph flowStepGraph, Collection<MapReduceFlowStep> flowSteps ) { for( MapReduceFlowStep flowStep : flowSteps ) flowStepGraph.addVertex( flowStep ); flowStepGraph.bindEdges(); return flowStepGraph; } }
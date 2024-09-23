public class HadoopFlowProcess extends FlowProcess<JobConf> implements MapRed { final JobConf jobConf; private final boolean isMapper; Reporter reporter = Reporter.NULL; private OutputCollector outputCollector; public HadoopFlowProcess() { this.jobConf = new JobConf(); this.isMapper = true; } public HadoopFlowProcess( Configuration jobConf ) { this( new JobConf( jobConf ) ); } public HadoopFlowProcess( JobConf jobConf ) { this.jobConf = jobConf; this.isMapper = true; } public HadoopFlowProcess( FlowSession flowSession, JobConf jobConf ) { super( flowSession ); this.jobConf = jobConf; this.isMapper = true; } public HadoopFlowProcess( FlowSession flowSession, JobConf jobConf, boolean isMapper ) { super( flowSession ); this.jobConf = jobConf; this.isMapper = isMapper; } public HadoopFlowProcess( HadoopFlowProcess flowProcess, JobConf jobConf ) { super( flowProcess ); this.jobConf = jobConf; this.isMapper = flowProcess.isMapper(); this.reporter = flowProcess.getReporter(); } @Override public FlowProcess copyWith( JobConf jobConf ) { return new HadoopFlowProcess( this, jobConf ); } public JobConf getJobConf() { return jobConf; } @Override public JobConf getConfig() { return jobConf; } @Override public JobConf getConfigCopy() { return HadoopUtil.copyJobConf( jobConf ); } public boolean isMapper() { return isMapper; } public int getCurrentNumMappers() { return getJobConf().getNumMapTasks(); } public int getCurrentNumReducers() { return getJobConf().getNumReduceTasks(); } @Override public int getCurrentSliceNum() { return getJobConf().getInt( "mapred.task.partition", 0 ); } @Override public int getNumProcessSlices() { if( isMapper() ) return getCurrentNumMappers(); else return getCurrentNumReducers(); } public void setReporter( Reporter reporter ) { if( reporter == null ) this.reporter = Reporter.NULL; else this.reporter = reporter; } @Override public Reporter getReporter() { return reporter; } public void setOutputCollector( OutputCollector outputCollector ) { this.outputCollector = outputCollector; } public OutputCollector getOutputCollector() { return outputCollector; } @Override public Object getProperty( String key ) { return jobConf.get( key ); } @Override public Collection<String> getPropertyKeys() { Set<String> keys = new HashSet<String>(); for( Map.Entry<String, String> entry : jobConf ) keys.add( entry.getKey() ); return Collections.unmodifiableSet( keys ); } @Override public Object newInstance( String className ) { if( className == null || className.isEmpty() ) return null; try { Class type = (Class) HadoopFlowProcess.class.getClassLoader().loadClass( className.toString() ); return ReflectionUtils.newInstance( type, jobConf ); } catch( ClassNotFoundException exception ) { throw new CascadingException( "unable to load class: " + className.toString(), exception ); } } @Override public void keepAlive() { getReporter().progress(); } @Override public void increment( Enum counter, long amount ) { getReporter().incrCounter( counter, amount ); } @Override public void increment( String group, String counter, long amount ) { getReporter().incrCounter( group, counter, amount ); } @Override public long getCounterValue( Enum counter ) { return getReporter().getCounter( counter ).getValue(); } @Override public long getCounterValue( String group, String counter ) { return getReporter().getCounter( group, counter ).getValue(); } @Override public void setStatus( String status ) { getReporter().setStatus( status ); } @Override public boolean isCounterStatusInitialized() { return getReporter() != null; } @Override public TupleEntryIterator openTapForRead( Tap tap ) throws IOException { return tap.openForRead( this ); } @Override public TupleEntryCollector openTapForWrite( Tap tap ) throws IOException { return tap.openForWrite( this, null ); } @Override public TupleEntryCollector openTrapForWrite( Tap trap ) throws IOException { JobConf jobConf = HadoopUtil.copyJobConf( getJobConf() ); int stepNum = jobConf.getInt( "cascading.flow.step.num", 0 ); String partname; if( jobConf.getBoolean( "mapred.task.is.map", true ) ) partname = String.format( "-m-%05d-", stepNum ); else partname = String.format( "-r-%05d-", stepNum ); jobConf.set( "cascading.tapcollector.partname", "%s%spart" + partname + "%05d" ); return trap.openForWrite( new HadoopFlowProcess( this, jobConf ), null ); } @Override public TupleEntryCollector openSystemIntermediateForWrite() throws IOException { return new TupleEntryCollector( Fields.size( 2 ) ) { @Override protected void collect( TupleEntry tupleEntry ) { try { getOutputCollector().collect( tupleEntry.getObject( 0 ), tupleEntry.getObject( 1 ) ); } catch( IOException exception ) { throw new CascadingException( "failed collecting key and value", exception ); } } }; } @Override public <C> C copyConfig( C config ) { return HadoopUtil.copyJobConf( config ); } @Override public <C> Map<String, String> diffConfigIntoMap( C defaultConfig, C updatedConfig ) { return HadoopUtil.getConfig( (Configuration) defaultConfig, (Configuration) updatedConfig ); } @Override public JobConf mergeMapIntoConfig( JobConf defaultConfig, Map<String, String> map ) { return HadoopUtil.mergeConf( defaultConfig, map, false ); } }
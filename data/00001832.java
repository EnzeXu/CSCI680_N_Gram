public class FlowMapper implements MapRunnable { private static final Logger LOG = LoggerFactory.getLogger( FlowMapper.class ); private FlowNode flowNode; private HadoopMapStreamGraph streamGraph; private HadoopFlowProcess currentProcess; public FlowMapper() { } @Override public void configure( JobConf jobConf ) { try { HadoopUtil.initLog4j( jobConf ); LOG.info( "cascading version: {}", jobConf.get( "cascading.version", "" ) ); LOG.info( "child jvm opts: {}", jobConf.get( "mapred.child.java.opts", "" ) ); currentProcess = new HadoopFlowProcess( new FlowSession(), jobConf, true ); String mapNodeState = jobConf.getRaw( "cascading.flow.step.node.map" ); if( mapNodeState == null ) mapNodeState = readStateFromDistCache( jobConf, jobConf.get( FlowStep.CASCADING_FLOW_STEP_ID ), "map" ); flowNode = deserializeBase64( mapNodeState, jobConf, BaseFlowNode.class ); LOG.info( "flow node id: {}, ordinal: {}", flowNode.getID(), flowNode.getOrdinal() ); Tap source = Flows.getTapForID( flowNode.getSourceTaps(), jobConf.get( "cascading.step.source" ) ); streamGraph = new HadoopMapStreamGraph( currentProcess, flowNode, source ); for( Duct head : streamGraph.getHeads() ) LOG.info( "sourcing from: " + ( (ElementDuct) head ).getFlowElement() ); for( Duct tail : streamGraph.getTails() ) LOG.info( "sinking to: " + ( (ElementDuct) tail ).getFlowElement() ); for( Tap trap : flowNode.getTraps() ) LOG.info( "trapping to: " + trap ); logMemory( LOG, "flow node id: " + flowNode.getID() + ", mem on start" ); } catch( Throwable throwable ) { reportIfLocal( throwable ); if( throwable instanceof CascadingException ) throw (CascadingException) throwable; throw new FlowException( "internal error during mapper configuration", throwable ); } } @Override public void run( RecordReader input, OutputCollector output, Reporter reporter ) throws IOException { currentProcess.setReporter( reporter ); currentProcess.setOutputCollector( output ); streamGraph.prepare(); long processBeginTime = System.currentTimeMillis(); currentProcess.increment( SliceCounters.Process_Begin_Time, processBeginTime ); currentProcess.increment( StepCounters.Process_Begin_Time, processBeginTime ); SourceStage streamedHead = streamGraph.getStreamedHead(); Iterator<Duct> iterator = streamGraph.getHeads().iterator(); try { try { while( iterator.hasNext() ) { Duct next = iterator.next(); if( next != streamedHead ) ( (SourceStage) next ).run( null ); } streamedHead.run( input ); } catch( OutOfMemoryError error ) { throw error; } catch( IOException exception ) { reportIfLocal( exception ); throw exception; } catch( Throwable throwable ) { reportIfLocal( throwable ); if( throwable instanceof CascadingException ) throw (CascadingException) throwable; throw new FlowException( "internal error during mapper execution", throwable ); } } finally { try { streamGraph.cleanup(); } finally { long processEndTime = System.currentTimeMillis(); currentProcess.increment( SliceCounters.Process_End_Time, processEndTime ); currentProcess.increment( SliceCounters.Process_Duration, processEndTime - processBeginTime ); currentProcess.increment( StepCounters.Process_End_Time, processEndTime ); currentProcess.increment( StepCounters.Process_Duration, processEndTime - processBeginTime ); String message = "flow node id: " + flowNode.getID(); logMemory( LOG, message + ", mem on close" ); logCounters( LOG, message + ", counter:", currentProcess ); } } } private void reportIfLocal( Throwable throwable ) { if( HadoopUtil.isLocal( currentProcess.getJobConf() ) ) HadoopFlowStepJob.reportLocalError( throwable ); } }
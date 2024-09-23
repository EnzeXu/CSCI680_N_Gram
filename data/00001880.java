public class LocalFlowStepJob extends FlowStepJob<Properties> { private final LocalStepRunner stackRunner; private Future<Throwable> future; public LocalFlowStepJob( ClientState clientState, LocalFlowProcess flowProcess, LocalFlowStep flowStep ) { super( clientState, flowStep.getConfig(), flowStep, 200, 1000, 1000 * 60 ); flowProcess.setStepStats( (LocalStepStats) this.flowStepStats ); this.stackRunner = new LocalStepRunner( flowProcess, flowStep ); } @Override protected FlowStepStats createStepStats( ClientState clientState ) { return new LocalStepStats( flowStep, clientState ); } @Override protected boolean isRemoteExecution() { return false; } @Override protected String internalJobId() { return "flow"; } @Override protected void internalNonBlockingStart() throws IOException { ExecutorService executors = Executors.newFixedThreadPool( 1 ); future = executors.submit( stackRunner ); executors.shutdown(); } @Override protected void updateNodeStatus( FlowNodeStats flowNodeStats ) { } @Override protected boolean internalIsStartedRunning() { return future != null; } @Override protected boolean internalNonBlockingIsComplete() throws IOException { return stackRunner.isCompleted(); } @Override protected Throwable getThrowable() { return stackRunner.getThrowable(); } @Override protected boolean internalNonBlockingIsSuccessful() throws IOException { return stackRunner.isSuccessful(); } @Override protected void internalBlockOnStop() throws IOException { stackRunner.blockUntilStopped(); } @Override protected void dumpDebugInfo() { } }
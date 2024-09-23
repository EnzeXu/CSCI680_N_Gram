public class FlowStepListenersTest extends TestCase { public FlowStepListenersTest ( ) { } public void testListeners ( ) { BaseFlowStep step = new BaseFlowStep ( "step1" , 0 ) { @ Override public Object createInitializedConfig ( FlowProcess fp , Object config ) { return null ; } @ Override public void clean ( Object config ) { } @ Override protected FlowStepJob createFlowStepJob ( ClientState clientState , FlowProcess fp , Object initializedStepConfig ) { return null ; } @ Override public Map < Object , Object > getConfigAsProperties ( ) { return Collections . emptyMap ( ) ; } @ Override public Set getTraps ( ) { return null ; } @ Override public Tap getTrap ( String string ) { return null ; } } ; FlowStepListener listener = new FlowStepListener ( ) { public void onStepStarting ( FlowStep flowStep ) { } public void onStepStopping ( FlowStep flowStep ) { } public void onStepCompleted ( FlowStep flowStep ) { } public boolean onStepThrowable ( FlowStep flowStep , Throwable throwable ) { return false ; } public void onStepRunning ( FlowStep flowStep ) { } } ; step . addListener ( listener ) ; assertTrue ( "no listener found" , step . hasListeners ( ) ) ; step . removeListener ( listener ) ; assertFalse ( "listener found" , step . hasListeners ( ) ) ; } }
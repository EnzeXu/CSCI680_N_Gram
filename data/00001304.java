public class ProcessFlowStats extends FlowStats { private ProcessWrapper processWrapper ; public ProcessFlowStats ( Flow flow , ClientState clientState , ProcessWrapper processWrapper ) { super ( flow , clientState ) ; this . processWrapper = processWrapper ; } @ Override public List < FlowStepStats > getFlowStepStats ( ) { return getChildrenInternal ( ) ; } @ Override public Collection getChildren ( ) { return getChildrenInternal ( ) ; } private List < FlowStepStats > getChildrenInternal ( ) { try { if ( !processWrapper . hasChildren ( ) ) { if ( processWrapper . hasCounters ( ) ) return Arrays . < FlowStepStats > asList ( new ProcessStepStats ( clientState , processWrapper . getCounters ( ) , new ProcessFlowStep ( processWrapper , 1 ) ) ) ; else return Collections . emptyList ( ) ; } List < FlowStepStats > childStepStats = new ArrayList < FlowStepStats > ( ) ; int counter = 0 ; for ( Object process : processWrapper . getChildren ( ) ) { ProcessWrapper childWrapper = new ProcessWrapper ( process ) ; if ( childWrapper . hasCounters ( ) ) { ProcessStepStats processStepStats = new ProcessStepStats ( clientState , childWrapper . getCounters ( ) , new ProcessFlowStep ( processWrapper , counter ) ) ; counter++ ; childStepStats . add ( processStepStats ) ; } } return childStepStats ; } catch ( ProcessException exception ) { throw new CascadingException ( exception ) ; } } @ Override public int getStepsCount ( ) { try { if ( !processWrapper . hasChildren ( ) ) return 1 ; return processWrapper . getChildren ( ) . size ( ) ; } catch ( ProcessException exception ) { throw new CascadingException ( exception ) ; } } }
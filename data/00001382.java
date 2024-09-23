public class ScopeResolver { private static final Logger LOG = LoggerFactory . getLogger ( ScopeResolver . class ) ; public static void resolveFields ( FlowElementGraph flowElementGraph ) { if ( flowElementGraph . isResolved ( ) ) throw new IllegalStateException ( "element graph already resolved" ) ; Iterator < FlowElement > iterator = ElementGraphs . getTopologicalIterator ( flowElementGraph ) ; while ( iterator . hasNext ( ) ) resolveFields ( flowElementGraph , iterator . next ( ) ) ; flowElementGraph . setResolved ( true ) ; } private static void resolveFields ( FlowElementGraph flowElementGraph , FlowElement source ) { if ( source instanceof Extent ) return ; Set < Scope > incomingScopes = flowElementGraph . incomingEdgesOf ( source ) ; Set < Scope > outgoingScopes = flowElementGraph . outgoingEdgesOf ( source ) ; List < FlowElement > flowElements = flowElementGraph . successorListOf ( source ) ; if ( flowElements . size ( ) == 0 ) throw new IllegalStateException ( "unable to find next elements in pipeline from : " + source . toString ( ) ) ; if ( ! ( source instanceof ScopedElement ) ) throw new IllegalStateException ( "flow element is not a scoped element : " + source . toString ( ) ) ; Scope outgoingScope = ( ( ScopedElement ) source ) . outgoingScopeFor ( incomingScopes ) ; if ( LOG . isDebugEnabled ( ) && outgoingScope != null ) { LOG . debug ( "for modifier : " + source ) ; if ( outgoingScope . getArgumentsSelector ( ) != null ) LOG . debug ( "setting outgoing arguments : " + outgoingScope . getArgumentsSelector ( ) ) ; if ( outgoingScope . getOperationDeclaredFields ( ) != null ) LOG . debug ( "setting outgoing declared : " + outgoingScope . getOperationDeclaredFields ( ) ) ; if ( outgoingScope . getKeySelectors ( ) != null ) LOG . debug ( "setting outgoing group : " + outgoingScope . getKeySelectors ( ) ) ; if ( outgoingScope . getOutValuesSelector ( ) != null ) LOG . debug ( "setting outgoing values : " + outgoingScope . getOutValuesSelector ( ) ) ; } for ( Scope scope : outgoingScopes ) scope . copyFields ( outgoingScope ) ; } }
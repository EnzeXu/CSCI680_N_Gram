public class IdentifierGraph extends TopologyGraph<String> { public IdentifierGraph( Flow... flows ) { super( flows ); } protected String getVertex( Flow flow, Tap tap ) { return tap.getFullIdentifier( flow.getConfig() ); } }
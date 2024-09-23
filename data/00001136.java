public class Neo4jJSONScheme extends Neo4jScheme { private static final Logger LOG = LoggerFactory.getLogger( Neo4jJSONScheme.class ); private final JSONGraphSpec graphSpec; public Neo4jJSONScheme( Fields sinkFields, JSONGraphSpec graphSpec ) { super( Fields.UNKNOWN, sinkFields ); this.graphSpec = graphSpec; if( !sinkFields.isDeclarator() || sinkFields.size() > 1 ) throw new IllegalArgumentException( "sink fields must be size one, got: " + sinkFields.print() ); } @Override public boolean isSource() { return false; } @Override public void sourceConfInit( FlowProcess<? extends Properties> flowProcess, Tap<Properties, Void, Session> tap, Properties conf ) { throw new UnsupportedOperationException(); } @Override public boolean source( FlowProcess<? extends Properties> flowProcess, SourceCall<Context, Void> sourceCall ) throws IOException { throw new UnsupportedOperationException(); } @Override public void sinkConfInit( FlowProcess<? extends Properties> flowProcess, Tap<Properties, Void, Session> tap, Properties conf ) { } @Override public void sinkPrepare( FlowProcess<? extends Properties> flowProcess, SinkCall<Context, Session> sinkCall ) throws IOException { sinkCall.setContext( new Context<>( new Neo4jJSONStatement( graphSpec ) ) ); } @Override public void sink( FlowProcess<? extends Properties> flowProcess, SinkCall<Context, Session> sinkCall ) throws IOException { Session session = sinkCall.getOutput(); Neo4jStatement<JsonNode> statement = sinkCall.getContext().statement; TupleEntry entry = sinkCall.getOutgoingEntry(); JsonNode node = (JsonNode) entry.getObject( 0 ); session.writeTransaction( tx -> { StatementResult result = statement.runStatement( tx, node ); if( LOG.isDebugEnabled() ) LOG.debug( "cypher results: {}", result.summary() ); return true; } ); } }
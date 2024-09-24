public class UnGroup extends BaseOperation implements Function { private static final Logger LOG = LoggerFactory.getLogger( UnGroup.class ); private Fields groupFieldSelector; private Fields[] resultFieldSelectors; private int size = 1; @ConstructorProperties({"groupSelector", "valueSelectors"}) public UnGroup( Fields groupSelector, Fields[] valueSelectors ) { if( valueSelectors == null || valueSelectors.length == 1 ) throw new IllegalArgumentException( "value selectors may not be empty" ); int size = valueSelectors[ 0 ].size(); for( int i = 1; i < valueSelectors.length; i++ ) { if( valueSelectors[ 0 ].size() != valueSelectors[ i ].size() ) throw new IllegalArgumentException( "all value selectors must be the same size" ); size = valueSelectors[ i ].size(); } this.numArgs = groupSelector.size() + size * valueSelectors.length; this.groupFieldSelector = groupSelector; this.resultFieldSelectors = Arrays.copyOf( valueSelectors, valueSelectors.length ); this.fieldDeclaration = Fields.size( groupSelector.size() + size ); } @ConstructorProperties({"fieldDeclaration", "groupSelector", "valueSelectors"}) public UnGroup( Fields fieldDeclaration, Fields groupSelector, Fields[] valueSelectors ) { super( fieldDeclaration ); if( valueSelectors == null || valueSelectors.length == 1 ) throw new IllegalArgumentException( "value selectors may not be empty" ); numArgs = groupSelector.size(); int selectorSize = -1; for( Fields resultFieldSelector : valueSelectors ) { numArgs += resultFieldSelector.size(); int fieldSize = groupSelector.size() + resultFieldSelector.size(); if( selectorSize != -1 && selectorSize != resultFieldSelector.size() ) throw new IllegalArgumentException( "all value selectors must be the same size, and this size plus group selector size must equal the declared field size" ); selectorSize = resultFieldSelector.size(); if( fieldDeclaration.size() != fieldSize ) throw new IllegalArgumentException( "all value selectors must be the same size, and this size plus group selector size must equal the declared field size" ); } this.groupFieldSelector = groupSelector; this.resultFieldSelectors = Arrays.copyOf( valueSelectors, valueSelectors.length ); } @ConstructorProperties({"fieldDeclaration", "groupSelector", "numValues"}) public UnGroup( Fields fieldDeclaration, Fields groupSelector, int numValues ) { super( fieldDeclaration ); this.groupFieldSelector = groupSelector; this.size = numValues; } @Property(name = "ungroupFieldSelector", visibility = Visibility.PRIVATE) @PropertyDescription("The fields to un-group.") public Fields getGroupFieldSelector() { return groupFieldSelector; } @Property(name = "resultFieldSelectors", visibility = Visibility.PRIVATE) @PropertyDescription("The result field selectors.") public Fields[] getResultFieldSelectors() { return Util.copy( resultFieldSelectors ); } public int getSize() { return size; } @Override public void operate( FlowProcess flowProcess, FunctionCall functionCall ) { if( resultFieldSelectors != null ) useResultSelectors( functionCall.getArguments(), functionCall.getOutputCollector() ); else useSize( functionCall.getArguments(), functionCall.getOutputCollector() ); } private void useSize( TupleEntry input, TupleEntryCollector outputCollector ) { LOG.debug( "using size: {}", size ); Tuple tuple = new Tuple( input.getTuple() ); Tuple group = tuple.remove( input.getFields(), groupFieldSelector ); for( int i = 0; i < tuple.size(); i = i + size ) { Tuple result = new Tuple( group ); result.addAll( tuple.get( Fields.offsetSelector( size, i ).getPos() ) ); outputCollector.add( result ); } } private void useResultSelectors( TupleEntry input, TupleEntryCollector outputCollector ) { LOG.debug( "using result selectors: {}", resultFieldSelectors.length ); for( Fields resultFieldSelector : resultFieldSelectors ) { Tuple group = input.selectTupleCopy( groupFieldSelector ); input.selectInto( resultFieldSelector, group ); outputCollector.add( group ); } } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof UnGroup ) ) return false; if( !super.equals( object ) ) return false; UnGroup unGroup = (UnGroup) object; if( size != unGroup.size ) return false; if( groupFieldSelector != null ? !groupFieldSelector.equals( unGroup.groupFieldSelector ) : unGroup.groupFieldSelector != null ) return false; if( !Arrays.equals( resultFieldSelectors, unGroup.resultFieldSelectors ) ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + ( groupFieldSelector != null ? groupFieldSelector.hashCode() : 0 ); result = 31 * result + ( resultFieldSelectors != null ? Arrays.hashCode( resultFieldSelectors ) : 0 ); result = 31 * result + size; return result; } }
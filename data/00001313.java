public class First extends ExtentBase { private final int firstN; public First() { super( Fields.ARGS ); this.firstN = 1; } @ConstructorProperties({"firstN"}) public First( int firstN ) { super( Fields.ARGS ); this.firstN = firstN; } @ConstructorProperties({"fieldDeclaration"}) public First( Fields fieldDeclaration ) { super( fieldDeclaration.size(), fieldDeclaration ); this.firstN = 1; } @ConstructorProperties({"fieldDeclaration", "firstN"}) public First( Fields fieldDeclaration, int firstN ) { super( fieldDeclaration.size(), fieldDeclaration ); this.firstN = firstN; } @ConstructorProperties({"fieldDeclaration", "ignoreTuples"}) public First( Fields fieldDeclaration, Tuple... ignoreTuples ) { super( fieldDeclaration, ignoreTuples ); this.firstN = 1; } @Property(name = "firstN", visibility = Visibility.PUBLIC) @PropertyDescription("The number of tuples to return.") public int getFirstN() { return firstN; } protected void performOperation( Tuple[] context, TupleEntry entry ) { if( context[ 0 ] == null ) context[ 0 ] = new Tuple(); if( context[ 0 ].size() < firstN ) context[ 0 ].add( entry.getTupleCopy() ); } @Override public void complete( FlowProcess flowProcess, AggregatorCall<Tuple[]> aggregatorCall ) { Tuple context = aggregatorCall.getContext()[ 0 ]; if( context == null ) return; for( Object tuple : context ) aggregatorCall.getOutputCollector().add( (Tuple) tuple ); } }
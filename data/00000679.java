public class VDUBuffer { public final static String ID = "$Id : VDUBuffer . java 503 2005-10-24 07 : 34 : 13Z marcus $" ; public final static int debug = 0 ; public int height , width ; public boolean [ ] update ; public char [ ] [ ] charArray ; public long [ ] [ ] charAttributes ; public int bufSize ; public int maxBufSize ; public int screenBase ; public int windowBase ; public int scrollMarker ; private int topMargin ; private int bottomMargin ; protected boolean showcursor = true ; protected int cursorX , cursorY ; public final static boolean SCROLL_UP = false ; public final static boolean SCROLL_DOWN = true ; public final static long NORMAL = 0x00 ; public final static long BOLD = 0x01 ; public final static long UNDERLINE = 0x02 ; public final static long INVERT = 0x04 ; public final static long LOW = 0x08 ; public final static long INVISIBLE = 0x10 ; public final static long FULLWIDTH = 0x20 ; public final static int COLOR_FG_SHIFT = 6 ; public final static int COLOR_BG_SHIFT = 31 ; public final static long COLOR = 0xffffffffffffc0L ; public final static long COLOR_FG = 0x7fffffc0L ; public final static long COLOR_BG = 0xffffff80000000L ; public final static int COLOR_RED_SHIFT = 16 ; public final static int COLOR_GREEN_SHIFT = 8 ; public final static int COLOR_BLUE_SHIFT = 0 ; public VDUBuffer ( int width , int height ) { setScreenSize ( width , height , false ) ; } public VDUBuffer ( ) { this ( 80 , 24 ) ; } public void putChar ( int c , int l , char ch ) { putChar ( c , l , ch , NORMAL ) ; } public void putChar ( int c , int l , char ch , long attributes ) { charArray [ screenBase + l ] [ c ] = ch ; charAttributes [ screenBase + l ] [ c ] = attributes ; if ( l < height ) update [ l + 1 ] = true ; } public char getChar ( int c , int l ) { return charArray [ screenBase + l ] [ c ] ; } public long getAttributes ( int c , int l ) { return charAttributes [ screenBase + l ] [ c ] ; } public void insertChar ( int c , int l , char ch , long attributes ) { System . arraycopy ( charArray [ screenBase + l ] , c , charArray [ screenBase + l ] , c + 1 , width - c - 1 ) ; System . arraycopy ( charAttributes [ screenBase + l ] , c , charAttributes [ screenBase + l ] , c + 1 , width - c - 1 ) ; putChar ( c , l , ch , attributes ) ; } public void deleteChar ( int c , int l ) { if ( c < width - 1 ) { System . arraycopy ( charArray [ screenBase + l ] , c + 1 , charArray [ screenBase + l ] , c , width - c - 1 ) ; System . arraycopy ( charAttributes [ screenBase + l ] , c + 1 , charAttributes [ screenBase + l ] , c , width - c - 1 ) ; } putChar ( width - 1 , l , ( char ) 0 ) ; } public void putString ( int c , int l , String s ) { putString ( c , l , s , NORMAL ) ; } public void putString ( int c , int l , String s , long attributes ) { for ( int i = 0 ; i < s . length ( ) && c + i < width ; i++ ) putChar ( c + i , l , s . charAt ( i ) , attributes ) ; } public void insertLine ( int l ) { insertLine ( l , 1 , SCROLL_UP ) ; } public void insertLine ( int l , int n ) { insertLine ( l , n , SCROLL_UP ) ; } public void insertLine ( int l , boolean scrollDown ) { insertLine ( l , 1 , scrollDown ) ; } public synchronized void insertLine ( int l , int n , boolean scrollDown ) { char cbuf [ ] [ ] = null ; long abuf [ ] [ ] = null ; int offset = 0 ; int oldBase = screenBase ; int newScreenBase = screenBase ; int newWindowBase = windowBase ; int newBufSize = bufSize ; if ( l > bottomMargin ) return ; int top = ( l < topMargin ? 0 : ( l > bottomMargin ? ( bottomMargin + 1 < height ? bottomMargin + 1 : height - 1 ) : topMargin ) ) ; int bottom = ( l > bottomMargin ? height - 1 : ( l < topMargin ? ( topMargin > 0 ? topMargin - 1 : 0 ) : bottomMargin ) ) ; if ( scrollDown ) { if ( n > ( bottom - top ) ) n = ( bottom - top ) ; int size = bottom - l - ( n - 1 ) ; if ( size < 0 ) size = 0 ; cbuf = new char [ size ] [ width ] ; abuf = new long [ size ] [ width ] ; System . arraycopy ( charArray , oldBase + l , cbuf , 0 , bottom - l - ( n - 1 ) ) ; System . arraycopy ( charAttributes , oldBase + l , abuf , 0 , bottom - l - ( n - 1 ) ) ; System . arraycopy ( cbuf , 0 , charArray , oldBase + l + n , bottom - l - ( n - 1 ) ) ; System . arraycopy ( abuf , 0 , charAttributes , oldBase + l + n , bottom - l - ( n - 1 ) ) ; cbuf = charArray ; abuf = charAttributes ; } else { try { if ( n > ( bottom - top ) + 1 ) n = ( bottom - top ) + 1 ; if ( bufSize < maxBufSize ) { if ( bufSize + n > maxBufSize ) { offset = n - ( maxBufSize - bufSize ) ; scrollMarker += offset ; newBufSize = maxBufSize ; newScreenBase = maxBufSize - height - 1 ; newWindowBase = screenBase ; } else { scrollMarker += n ; newScreenBase += n ; newWindowBase += n ; newBufSize += n ; } cbuf = new char [ newBufSize ] [ width ] ; abuf = new long [ newBufSize ] [ width ] ; } else { offset = n ; cbuf = charArray ; abuf = charAttributes ; } if ( oldBase > 0 ) { System . arraycopy ( charArray , offset , cbuf , 0 , oldBase - offset ) ; System . arraycopy ( charAttributes , offset , abuf , 0 , oldBase - offset ) ; } if ( top > 0 ) { System . arraycopy ( charArray , oldBase , cbuf , newScreenBase , top ) ; System . arraycopy ( charAttributes , oldBase , abuf , newScreenBase , top ) ; } if ( oldBase > = 0 ) { System . arraycopy ( charArray , oldBase + top , cbuf , oldBase - offset , n ) ; System . arraycopy ( charAttributes , oldBase + top , abuf , oldBase - offset , n ) ; } System . arraycopy ( charArray , oldBase + top + n , cbuf , newScreenBase + top , l - top - ( n - 1 ) ) ; System . arraycopy ( charAttributes , oldBase + top + n , abuf , newScreenBase + top , l - top - ( n - 1 ) ) ; if ( l < height - 1 ) { System . arraycopy ( charArray , oldBase + l + 1 , cbuf , newScreenBase + l + 1 , ( height - 1 ) - l ) ; System . arraycopy ( charAttributes , oldBase + l + 1 , abuf , newScreenBase + l + 1 , ( height - 1 ) - l ) ; } } catch ( ArrayIndexOutOfBoundsException e ) { System . err . println ( "*** Error while scrolling up : " ) ; System . err . println ( "--- BEGIN STACK TRACE ---" ) ; e . printStackTrace ( ) ; System . err . println ( "--- END STACK TRACE ---" ) ; System . err . println ( "bufSize=" + bufSize + " , maxBufSize=" + maxBufSize ) ; System . err . println ( "top=" + top + " , bottom=" + bottom ) ; System . err . println ( "n=" + n + " , l=" + l ) ; System . err . println ( "screenBase=" + screenBase + " , windowBase=" + windowBase ) ; System . err . println ( "newScreenBase=" + newScreenBase + " , newWindowBase=" + newWindowBase ) ; System . err . println ( "oldBase=" + oldBase ) ; System . err . println ( "size . width=" + width + " , size . height=" + height ) ; System . err . println ( "abuf . length=" + abuf . length + " , cbuf . length=" + cbuf . length ) ; System . err . println ( "*** done dumping debug information" ) ; } } scrollMarker -= n ; for ( int i = 0 ; i < n ; i++ ) { cbuf [ ( newScreenBase + l ) + ( scrollDown ? i : -i ) ] = new char [ width ] ; Arrays . fill ( cbuf [ ( newScreenBase + l ) + ( scrollDown ? i : -i ) ] , ' ' ) ; abuf [ ( newScreenBase + l ) + ( scrollDown ? i : -i ) ] = new long [ width ] ; } charArray = cbuf ; charAttributes = abuf ; screenBase = newScreenBase ; windowBase = newWindowBase ; bufSize = newBufSize ; if ( scrollDown ) markLine ( l , bottom - l + 1 ) ; else markLine ( top , l - top + 1 ) ; display . updateScrollBar ( ) ; } public void deleteLine ( int l ) { int bottom = ( l > bottomMargin ? height - 1 : ( l < topMargin?topMargin : bottomMargin + 1 ) ) ; int numRows = bottom - l - 1 ; char [ ] discardedChars = charArray [ screenBase + l ] ; long [ ] discardedAttributes = charAttributes [ screenBase + l ] ; if ( numRows > 0 ) { System . arraycopy ( charArray , screenBase + l + 1 , charArray , screenBase + l , numRows ) ; System . arraycopy ( charAttributes , screenBase + l + 1 , charAttributes , screenBase + l , numRows ) ; } int newBottomRow = screenBase + bottom - 1 ; charArray [ newBottomRow ] = discardedChars ; charAttributes [ newBottomRow ] = discardedAttributes ; Arrays . fill ( charArray [ newBottomRow ] , ' ' ) ; Arrays . fill ( charAttributes [ newBottomRow ] , 0 ) ; markLine ( l , bottom - l ) ; } public void deleteArea ( int c , int l , int w , int h , long curAttr ) { int endColumn = c + w ; int targetRow = screenBase + l ; for ( int i = 0 ; i < h && l + i < height ; i++ ) { Arrays . fill ( charAttributes [ targetRow ] , c , endColumn , curAttr ) ; Arrays . fill ( charArray [ targetRow ] , c , endColumn , ' ' ) ; targetRow++ ; } markLine ( l , h ) ; } public void deleteArea ( int c , int l , int w , int h ) { deleteArea ( c , l , w , h , 0 ) ; } public void showCursor ( boolean doshow ) { showcursor = doshow ; } public boolean isCursorVisible ( ) { return showcursor ; } public void setCursorPosition ( int c , int l ) { cursorX = c ; cursorY = l ; } public int getCursorColumn ( ) { return cursorX ; } public int getCursorRow ( ) { return cursorY ; } public void setWindowBase ( int line ) { if ( line > screenBase ) line = screenBase ; else if ( line < 0 ) line = 0 ; windowBase = line ; update [ 0 ] = true ; redraw ( ) ; } public int getWindowBase ( ) { return windowBase ; } public void setMargins ( int l1 , int l2 ) { if ( l1 > l2 ) return ; if ( l1 < 0 ) l1 = 0 ; if ( l2 > = height ) l2 = height - 1 ; topMargin = l1 ; bottomMargin = l2 ; } public void setTopMargin ( int l ) { if ( l > bottomMargin ) { topMargin = bottomMargin ; bottomMargin = l ; } else topMargin = l ; if ( topMargin < 0 ) topMargin = 0 ; if ( bottomMargin > = height ) bottomMargin = height - 1 ; } public int getTopMargin ( ) { return topMargin ; } public void setBottomMargin ( int l ) { if ( l < topMargin ) { bottomMargin = topMargin ; topMargin = l ; } else bottomMargin = l ; if ( topMargin < 0 ) topMargin = 0 ; if ( bottomMargin > = height ) bottomMargin = height - 1 ; } public int getBottomMargin ( ) { return bottomMargin ; } public void setBufferSize ( int amount ) { if ( amount < height ) amount = height ; if ( amount < maxBufSize ) { char cbuf [ ] [ ] = new char [ amount ] [ width ] ; long abuf [ ] [ ] = new long [ amount ] [ width ] ; int copyStart = bufSize - amount < 0 ? 0 : bufSize - amount ; int copyCount = bufSize - amount < 0 ? bufSize : amount ; if ( charArray != null ) System . arraycopy ( charArray , copyStart , cbuf , 0 , copyCount ) ; if ( charAttributes != null ) System . arraycopy ( charAttributes , copyStart , abuf , 0 , copyCount ) ; charArray = cbuf ; charAttributes = abuf ; bufSize = copyCount ; screenBase = bufSize - height ; windowBase = screenBase ; } maxBufSize = amount ; update [ 0 ] = true ; redraw ( ) ; } public int getBufferSize ( ) { return bufSize ; } public int getMaxBufferSize ( ) { return maxBufSize ; } public void setScreenSize ( int w , int h , boolean broadcast ) { char cbuf [ ] [ ] ; long abuf [ ] [ ] ; int maxSize = bufSize ; int oldAbsR = screenBase + getCursorRow ( ) ; if ( w < 1 || h < 1 ) return ; if ( debug > 0 ) System . err . println ( "VDU : screen size [ " + w + " , " + h + " ] " ) ; if ( h > maxBufSize ) maxBufSize = h ; if ( h > bufSize ) { bufSize = h ; screenBase = 0 ; windowBase = 0 ; } if ( windowBase + h > = bufSize ) windowBase = bufSize - h ; if ( screenBase + h > = bufSize ) screenBase = bufSize - h ; cbuf = new char [ bufSize ] [ w ] ; abuf = new long [ bufSize ] [ w ] ; for ( int i = 0 ; i < bufSize ; i++ ) { Arrays . fill ( cbuf [ i ] , ' ' ) ; } if ( bufSize < maxSize ) maxSize = bufSize ; int rowLength ; if ( charArray != null && charAttributes != null ) { for ( int i = 0 ; i < maxSize && charArray [ i ] != null ; i++ ) { rowLength = charArray [ i ] . length ; System . arraycopy ( charArray [ i ] , 0 , cbuf [ i ] , 0 , w < rowLength ? w : rowLength ) ; System . arraycopy ( charAttributes [ i ] , 0 , abuf [ i ] , 0 , w < rowLength ? w : rowLength ) ; } } int C = getCursorColumn ( ) ; if ( C < 0 ) C = 0 ; else if ( C > = w ) C = w - 1 ; int R = getCursorRow ( ) ; if ( R + screenBase < = oldAbsR ) R = oldAbsR - screenBase ; if ( R < 0 ) R = 0 ; else if ( R > = h ) R = h - 1 ; setCursorPosition ( C , R ) ; charArray = cbuf ; charAttributes = abuf ; width = w ; height = h ; topMargin = 0 ; bottomMargin = h - 1 ; update = new boolean [ h + 1 ] ; update [ 0 ] = true ; } public int getRows ( ) { return height ; } public int getColumns ( ) { return width ; } public void markLine ( int l , int n ) { for ( int i = 0 ; ( i < n ) && ( l + i < height ) ; i++ ) update [ l + i + 1 ] = true ; } protected VDUDisplay display ; public void setDisplay ( VDUDisplay display ) { this . display = display ; } protected void redraw ( ) { if ( display != null ) display . redraw ( ) ; } }
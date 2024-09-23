public class ColorsActivity extends AppCompatActivity implements OnItemClickListener, OnColorChangedListener, OnItemSelectedListener { private GridView mColorGrid; private Spinner mFgSpinner; private Spinner mBgSpinner; private int mColorScheme; private int[] mColorList; private ColorStorage mHostDb; private int mCurrentColor = 0; private int[] mDefaultColors; @Override protected void onCreate(Bundle savedInstanceState) { super.onCreate(savedInstanceState); setContentView(R.layout.act_colors); mColorScheme = HostDatabase.DEFAULT_COLOR_SCHEME; mHostDb = HostDatabase.get(this); mColorList = mHostDb.getColorsForScheme(mColorScheme); mDefaultColors = mHostDb.getDefaultColorsForScheme(mColorScheme); mColorGrid = findViewById(R.id.color_grid); mColorGrid.setAdapter(new ColorsAdapter(true)); mColorGrid.setOnItemClickListener(this); mColorGrid.setSelection(0); mFgSpinner = findViewById(R.id.fg); mFgSpinner.setAdapter(new ColorsAdapter(false, R.string.colors_fg_label)); mFgSpinner.setSelection(mDefaultColors[0]); mFgSpinner.setOnItemSelectedListener(this); mBgSpinner = findViewById(R.id.bg); mBgSpinner.setAdapter(new ColorsAdapter(false, R.string.color_bg_label)); mBgSpinner.setSelection(mDefaultColors[1]); mBgSpinner.setOnItemSelectedListener(this); } @Override protected void onDestroy() { super.onDestroy(); if (mHostDb != null) { mHostDb = null; } } @Override protected void onResume() { super.onResume(); if (mHostDb == null) { mHostDb = HostDatabase.get(this); } } private class ColorsAdapter extends BaseAdapter { private final boolean mSquareViews; private final int mResourceLabel; public ColorsAdapter(boolean squareViews) { this(squareViews, -1); } public ColorsAdapter(boolean squareViews, int resourceLabel) { mSquareViews = squareViews; mResourceLabel = resourceLabel; } @Override public View getView(int position, View convertView, ViewGroup parent) { ColorView c; if (convertView == null) { c = new ColorView(ColorsActivity.this, mResourceLabel, mSquareViews); } else { c = (ColorView) convertView; } c.setColor(mColorList[position]); c.setNumber(position + 1); return c; } @Override public int getCount() { return mColorList.length; } @Override public Object getItem(int position) { return mColorList[position]; } @Override public long getItemId(int position) { return position; } } private class ColorView extends View { private static final float FONT_SIZE_DP = 20f; private static final float MARGIN_DP = 10f; private final boolean mSquare; private final int mResourceLabel; private final NumberFormat mNumberFormatter; private Paint mTextPaint; private Paint mShadowPaint; private int mBackgroundColor; private String mText; private int mAscent; private int mWidthCenter; private int mHeightCenter; public ColorView(Context context, int resourceLabel, boolean square) { super(context); mSquare = square; mResourceLabel = resourceLabel; mNumberFormatter = NumberFormat.getIntegerInstance(getContext().getResources().getConfiguration().locale); DisplayMetrics metrics = context.getResources().getDisplayMetrics(); mTextPaint = new Paint(); mTextPaint.setAntiAlias(true); mTextPaint.setTextSize((int) (metrics.density * FONT_SIZE_DP + 0.5f)); mTextPaint.setColor(0xFFFFFFFF); mTextPaint.setTextAlign(Paint.Align.CENTER); mShadowPaint = new Paint(mTextPaint); mShadowPaint.setStyle(Paint.Style.STROKE); mShadowPaint.setStrokeCap(Paint.Cap.ROUND); mShadowPaint.setStrokeJoin(Paint.Join.ROUND); mShadowPaint.setStrokeWidth(4f); mShadowPaint.setColor(0xFF000000); int marginPx = (int) (MARGIN_DP * metrics.density + 0.5f); setPadding(marginPx, marginPx, marginPx, marginPx); } public void setColor(int color) { mBackgroundColor = color; } public void setNumber(int number) { if (mResourceLabel != -1) { mText = getContext().getResources().getString(mResourceLabel, number); } else { mText = mNumberFormatter.format(number); } } @Override protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) { int width = measureWidth(widthMeasureSpec); int height; if (mSquare) { height = width; } else { height = measureHeight(heightMeasureSpec); } mAscent = (int) mTextPaint.ascent(); mWidthCenter = width / 2; mHeightCenter = height / 2 - mAscent / 2; setMeasuredDimension(width, height); } private int measureWidth(int measureSpec) { int result; int specMode = MeasureSpec.getMode(measureSpec); int specSize = MeasureSpec.getSize(measureSpec); if (specMode == MeasureSpec.EXACTLY) { result = specSize; } else { result = (int) mTextPaint.measureText(mText) + getPaddingLeft() + getPaddingRight(); if (specMode == MeasureSpec.AT_MOST) { result = Math.min(result, specSize); } } return result; } private int measureHeight(int measureSpec) { int result; int specMode = MeasureSpec.getMode(measureSpec); int specSize = MeasureSpec.getSize(measureSpec); mAscent = (int) mTextPaint.ascent(); if (specMode == MeasureSpec.EXACTLY) { result = specSize; } else { result = (int) (-mAscent + mTextPaint.descent()) + getPaddingTop() + getPaddingBottom(); if (specMode == MeasureSpec.AT_MOST) { result = Math.min(result, specSize); } } return result; } @Override protected void onDraw(Canvas canvas) { super.onDraw(canvas); canvas.drawColor(mBackgroundColor); canvas.drawText(mText, mWidthCenter, mHeightCenter, mShadowPaint); canvas.drawText(mText, mWidthCenter, mHeightCenter, mTextPaint); } } private void editColor(int colorNumber) { mCurrentColor = colorNumber; new UberColorPickerDialog(this, this, mColorList[colorNumber]).show(); } @Override public void onItemClick(AdapterView<?> parent, View view, int position, long id) { editColor(position); } @Override public void onNothingSelected(AdapterView<?> arg0) { } @Override public void colorChanged(int value) { mHostDb.setGlobalColor(mCurrentColor, value); mColorList[mCurrentColor] = value; mColorGrid.invalidateViews(); } @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) { boolean needUpdate = false; if (parent == mFgSpinner) { if (position != mDefaultColors[0]) { mDefaultColors[0] = position; needUpdate = true; } } else if (parent == mBgSpinner) { if (position != mDefaultColors[1]) { mDefaultColors[1] = position; needUpdate = true; } } if (needUpdate) { mHostDb.setDefaultColorsForScheme(mColorScheme, mDefaultColors[0], mDefaultColors[1]); } } @Override public boolean onCreateOptionsMenu(Menu menu) { super.onCreateOptionsMenu(menu); MenuItem reset = menu.add(R.string.menu_colors_reset); reset.setAlphabeticShortcut('r'); reset.setNumericShortcut('1'); reset.setIcon(android.R.drawable.ic_menu_revert); reset.setOnMenuItemClickListener(new OnMenuItemClickListener() { @Override public boolean onMenuItemClick(MenuItem arg0) { for (int i = 0; i < Colors.defaults.length; i++) { if (mColorList[i] != Colors.defaults[i]) { mHostDb.setGlobalColor(i, Colors.defaults[i]); mColorList[i] = Colors.defaults[i]; } } mColorGrid.invalidateViews(); mFgSpinner.setSelection(HostDatabase.DEFAULT_FG_COLOR); mBgSpinner.setSelection(HostDatabase.DEFAULT_BG_COLOR); mHostDb.setDefaultColorsForScheme(HostDatabase.DEFAULT_COLOR_SCHEME, HostDatabase.DEFAULT_FG_COLOR, HostDatabase.DEFAULT_BG_COLOR); return true; } }); return true; } }
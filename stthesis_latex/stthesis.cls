% Support class for diploma theses and similar works at the chair of Software Technology at Technische Universität Dresden
% (c) 2011 Jan Reimann, based on stthesis package by Steffen Zschaler
%
% TODO
% ----
% * better page layout with smaller margins?
%
% HISTORY
% -------
% 
% [2011-05-04]
%		* converted titlepage from Julia Schröter to package which can be used for setting TUD layout
%		* minor changes
% [2011-04-27]
%		* logo on title page
%		* front-, main-, backmatter with page numbering
% [2011-04-26] 
%		* Converted Package into class
%		* better language handling (english, german, ngerman)
% [2007-04-23] Added new german spelling of daß when using option ngerman instead of german
%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\newcommand{\stthesis}{stthesis}
\ProvidesClass{\stthesis}[2011/04/27 \stthesis{} class for writing student theses at the chair of Softwaretechnologie of the Technische Universitaet Dresden]

% is based on KOMA-scrbook
\LoadClass[a4paper,twoside,BCOR=12mm,pointlessnumbers]{scrbook}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Options for declaring the type of thesis
\DeclareOption{da}{\gdef\@type{\type@DA}} % Diploma Thesis
\DeclareOption{gb}{\gdef\@type{\type@GB}} % Großer Beleg
\DeclareOption{bs}{\gdef\@type{\type@BS}} % Bachelor
\DeclareOption{ms}{\gdef\@type{\type@MS}} % Master's Thesis

% Options for language
\newif\if@germanLang\@germanLangfalse
\newif\if@ngermanLang\@ngermanLangfalse
\newif\if@englishLang\@englishLangfalse
\DeclareOption{english}{\@englishLangtrue\@germanLangfalse}
\DeclareOption{german}{\@germanLangtrue\@ngermanLangfalse}
\DeclareOption{ngerman}{\@germanLangtrue\@ngermanLangtrue}

% Defaults
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ExecuteOptions{da,ngerman}

\ProcessOptions

% Load some Packages
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage[left=30mm,right=20mm,top=30mm,bottom=40mm]{geometry}
% needed for correct print output
\RequirePackage[cmyk, table]{xcolor}


\RequirePackage[
	pdftoolbar=true,
	bookmarksopen,
	bookmarksnumbered=false,
	bookmarksopenlevel=1,
	pdfdisplaydoctitle,	
    colorlinks,%
    citecolor=black,%
    filecolor=black,%
    linkcolor=black,%
    urlcolor=black
]{hyperref}
% needed for correct indentation
\RequirePackage{xspace}

% Set Languages
\if@englishLang
	\RequirePackage[english]{babel}
\else
	\if@ngermanLang
		\RequirePackage[ngerman]{babel}
	\else
			\if@germanLang
			\RequirePackage[german]{babel}
		\fi
	\fi
\fi


% Make sure the confirmation of independence is set at the end of the document
\AtEndDocument{\@independenceStatement}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We don't support footnotes on the title page
\let\thanks\relax
\def\@birthday{\ClassWarning{\stthesis}{No birthday specified! Use \string\birthday\space command.}}
\def\birthday#1{\gdef\@birthday{#1}}

\def\@birthplace{\ClassWarning{\stthesis}{No birthplace specified! Use \string\birthplace\space command.}}
\def\birthplace#1{\gdef\@birthplace{#1}}

\def\@supervisor{\ClassWarning{\stthesis}{No supervisor specified! Use \string\supervisor\space command.}}
\def\supervisor#1{\gdef\@supervisor{#1}}

\def\@hsl{Prof. Dr. rer. nat. habil. Uwe A\ss mann}
\def\hsl#1{\gdef\@hsl{#1}}


% We overwrite \maketitle to always create a title page and to make it fit our layout
\gdef\maketitle{%
  \begin{titlepage}%
    \vspace*{3cm}%
    \begin{center}%
      \Large{\@type}\\%
      \bigskip
      \huge{\textbf{\textsf{\@title}}}\\
      \vspace{2cm}
      \large{\s@authored} 
      \vfill
      \Large{\@author}\\
      \vfill
      \large{\s@born{\@birthday}{\@birthplace}}\\
      \vspace{1.5cm}%
      \vfill%
      \Large{Technische Universit\"{a}t Dresden}\\%
      \vfill%
      \large{Fakult\"{a}t Informatik}\\%
      \large{Institut f\"{u}r Software- und Multimediatechnik}\\%
      \large{Lehrstuhl Softwaretechnologie}\\%
      \vspace{20pt}
      \includegraphics[height=1.8cm]{tu_logo}\hspace{20pt}\includegraphics[height=2cm]{STgroupLogo}
      \vspace{10pt}
    \end{center}%
    %\vspace{3cm}%
    \large{\s@supervisor: \@supervisor\\%
           \s@hsl: \@hsl\\%
           \\%
           \s@submitted{\@date}}
	% some metadata
	\hypersetup{
		pdftitle={\@title},
		pdfauthor={\@author},
		pdfsubject={\@type}
	}
  \end{titlepage}%
  \global\let\maketitle\relax
  \global\let\@author\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
	\global\let\@birthday\@empty
	\global\let\birthday\relax
	\global\let\@birthplace\@empty
	\global\let\birthplace\relax
	\global\let\@supervisor\@empty
	\global\let\supervisor\relax
	\global\let\@hsl\@empty
	\global\let\hsl\relax
}

% Final Statement of Copyright
\def\@independenceStatement{%
  \newpage
  \pagestyle{empty}
  \section*{\if@germanLang Erkl\"{a}rung\else Confirmation\fi}%
    %
    \if@germanLang %
      Ich erkl\"{a}re, \if@ngermanLang dass\else da\ss\fi{} ich die vorliegende Arbeit \if@ngermanLang selbstst\"{a}ndig\else selbst\"{a}ndig\fi, unter Angabe aller Zitate und nur unter Verwendung der
      angegebenen Literatur und Hilfsmittel angefertigt habe.
    \else %
      I confirm that I independently prepared the thesis and that I used only the references and auxiliary means indicated in the
      thesis.
    \fi %
    \\
    \\
    \\
    \\
    \\
    \\
    \\
    Dresden, \if@germanLang den \fi \@date
}

% Strings to be used for the type of thesis
\def\type@DA{\if@germanLang Diplomarbeit\else Diploma Thesis\fi}
\def\type@GB{Gro\ss er Beleg}
\def\type@BS{\if@germanLang Bakkalaureatsarbeit\else Bachelor's Thesis\fi}
\def\type@MS{\if@germanLang Masterarbeit\else Master's Thesis\fi}

% Other Strings
\if@germanLang
  \def\s@authored{bearbeitet von}
  \def\s@born#1#2{geboren am #1 in #2}
  \def\s@supervisor{Betreuer}
  \def\s@hsl{Hochschullehrer}
  \def\s@submitted#1{Eingereicht am\xspace#1}
\else
  \def\s@authored{submitted by}
  \def\s@born#1#2{born #1 in #2}
  \def\s@supervisor{Supervisor}
  \def\s@hsl{Professor}
  \def\s@submitted#1{Submitted\xspace#1}
\fi

\let\@oldtoc\tableofcontents
\newcommand{\@mytoc}{\@oldtoc\mainmatter}
\let\tableofcontents\@mytoc

\renewcommand{\backmatter}{
	\newpage
	\pagenumbering{roman}
}

% some things to do at the beginning
\AtBeginDocument{
	\pagenumbering{Roman}
}


\endinput
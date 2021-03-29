#!/bin/bash
# Check returned versions for:
#
# * Install from git zip archive
# * Standard install
# * Local editable install.

PYTHON=${PYTHON:-python3}

VER_CHK_TMP=${PWD}/version_check_tmp
VER_FILE=${VER_CHK_TMP}/version_checks.txt
VENV_DIR=${VER_CHK_TMP}/venv
ZIP_FNAME=${VER_CHK_TMP}/project.zip
ZIP_DIR=${VER_CHK_TMP}/unpacked_zip

function mk_newdir {
    rm -rf $1
    mkdir $1
}

function extra_install {
    # Extra steps prior to pip install.
    # Will differ by package.
    pip install numpy
}

function install_show {
    local pkg_name=$1
    [ -z "$pkg_name" ] && (echo "Need package name" && exit 1)
    shift
    local inst_type=$1
    [ -z "$inst_type" ] && (echo "Need installation type" && exit 2)
    shift
    mk_newdir ${VENV_DIR}
    ${PYTHON} -m virtualenv ${VENV_DIR}
    ( . ${VENV_DIR}/bin/activate && \
        extra_install && \
        pip install $@ && \
        local pkg_ver=$(python -c "import $pkg_name; print(${pkg_name}.__version__)") && \
        echo "${pkg_name} - ${inst_type}: ${pkg_ver}" >> ${VER_FILE} && \
        deactivate )
}

mk_newdir ${VER_CHK_TMP}
cat << EOF > ${VER_FILE}
########
Versions
########

EOF

# Git zip archive
git archive --format zip -o $ZIP_FNAME HEAD
mk_newdir $ZIP_DIR
(cd $ZIP_DIR && unzip $ZIP_FNAME)
install_show nipy zip ${ZIP_DIR}

# Standard install
install_show nipy install .

# Local editable install
install_show nipy editable -e .

cat ${VER_FILE}
